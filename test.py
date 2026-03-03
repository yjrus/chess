import os
import time
import random
from collections import deque

import chess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chess_game import Chess
from model import ChessModel
from chess_data_processing import (
    MCTS,
    MCTS_simulations,
    boardToTensor,
    movesMask,
    codeToPromotion,
    promotionIndex,
    dirichletNoise,
    puct,
    selectMoveWithTemperature
)


weight_path = "drive/MyDrive/Chess/checkpoints/chess_model_iter_" if input("Colab? (y/n)").lower() == 'y' else "checkpoints/chess_model_iter_"


def selfPlayGame(model,
                  temperature: float = 1.0,
                  max_simulations: int = 100,
                  root_reuse: bool = True):
    """
    Запускает одну игру self-play с MCTS + модель.
    Возвращает список данных (state, policy, player) и финальный результат z.
    """
    model.eval()
    board = Chess()
    root = MCTS(move=None)
    game_data = []

    move_number = 1
    half_move_count = 0

    print(f"\nНовая игра началась!")

    while not board.is_game_over():

        move = MCTS_simulations(root,
                                model,
                                board,
                                count=max_simulations)

        if move is None:
            break

        pi = torch.zeros(4184, dtype=torch.float32)

        for child in root.children:
            if child.move.promotion is None:
                idx = child.move.from_square * 64 + child.move.to_square
            else:
                idx = 4096 + promotionIndex(child.move)

            pi[idx] = child.N

        if pi.sum() > 0:
            pi /= pi.sum()

        state_tensor = torch.cat([boardToTensor(board) for _ in range(4)], dim=0)
        player = 1 if board.turn == chess.WHITE else -1
        game_data.append((state_tensor, pi.clone(), player))

        board.push(move)
        half_move_count += 1

        print(f"Ход {move_number}: {move.uci()}")

        if half_move_count % 10 == 0:
            print(f"\n--- Доска после {half_move_count} полуходов ---")
            board.print_()
            print()

        move_number += 1

        if root_reuse:
            next_root = None

            for child in root.children:
                if child.move == move:
                    next_root = child
                    next_root.parent = None
                    break

            root = next_root if next_root else MCTS(move=None)

    result = board.result()

    if result == "1-0":
        z = 1
    elif result == "0-1":
        z = -1
    else:
        z = 0

    print(f"Игра завершена! Результат: {result}, всего ходов: {move_number-1}")

    return game_data, z



def generateSelfPlayGames(model,
                           n_games: int = 10,
                           temperature: float = 1.0,
                           max_simulations: int = 100,
                           root_reuse: bool = True,
                           device: str = 'cpu'):
    """
    Генерирует n_games игр с self-play и возвращает список данных для обучения.
    Каждый элемент: (state_tensor, policy_tensor, player, z)
    """
    model.to(device)
    dataset = []

    for game_idx in range(n_games):

        game_data, z = selfPlayGame(model,
                                      temperature=temperature,
                                      max_simulations=max_simulations,
                                      root_reuse=root_reuse)

        for state_tensor, pi, player in game_data:
            dataset.append((state_tensor.to(device),
                            pi.to(device),
                            player,
                            z * player))

        print(f"Game {game_idx+1}/{n_games} finished. Total positions: {len(game_data)}")

    return dataset



def trainOnDataset(model: ChessModel,
                    dataset,
                    batch_size: int = 32,
                    epochs: int = 1,
                    lr: float = 1e-3,
                    device: str = "cuda"):
    """
    Обучает модель на датасете (state, policy, value).
    Возвращает средний loss для мониторинга.
    """
    model.train()
    model.to(device)

    states = torch.stack([s for s, _, _, _ in dataset]).to(device)
    pis = torch.stack([p.flatten() for _, p, _, _ in dataset]).to(device)
    zs = torch.tensor([z for _, _, _, z in dataset],
                       dtype=torch.float32).to(device)

    tensor_dataset = TensorDataset(states, pis, zs)
    dataloader = DataLoader(tensor_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for batch_states, batch_pis, batch_zs in dataloader:
            optimizer.zero_grad()

            if batch_states.dim() == 3:
                batch_states = batch_states.unsqueeze(0)

            logits, values = model(batch_states)

            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(batch_pis * log_probs).sum(dim=1).mean()

            values = values.squeeze(1)
            value_loss = value_loss_fn(values, batch_zs)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item() * batch_states.size(0)
            total_value_loss += value_loss.item() * batch_states.size(0)

        n_samples = len(dataset)
        avg_policy_loss = total_policy_loss / n_samples
        avg_value_loss = total_value_loss / n_samples
        avg_total_loss = avg_policy_loss + avg_value_loss

        print(f"  Epoch {epoch+1}/{epochs} | Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | Total: {avg_total_loss:.4f}")

    return avg_total_loss



def mainTrainingLoop(model,
                      num_iterations: int = 50,
                      games_per_iteration: int = 5,
                      device=None):
    """
    Основной цикл обучения модели через self-play
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    total_positions_generated = 0
    best_loss = float('inf')

    print("=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ")
    print("=" * 60)
    print("Параметры:")
    print(f"- Итераций: {num_iterations}")
    print(f"- Игр за итерацию: {games_per_iteration}")
    print(f"- Устройство: {device}")
    print(f"- Модель: {model.__class__.__name__}")
    print(f"- Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    for iteration in range(num_iterations):
        print("=" * 60)
        print(f"ИТЕРАЦИЯ {iteration+1}/{num_iterations}")
        print("=" * 60)

        iteration_start_time = time.time()

        # Генерация self-play игр
        print("\nГенерация игр...")
        games_start_time = time.time()

        dataset = generateSelfPlayGames(model,
                                          n_games=games_per_iteration,
                                          device=device)

        games_time = time.time() - games_start_time
        positions_in_iteration = len(dataset)
        total_positions_generated += positions_in_iteration

        print(f" Сгенерировано игр: {games_per_iteration}")
        print(f" Позиций в итерации: {positions_in_iteration}")
        print(f" сего позиций сгенерировано: {total_positions_generated}")
        print(f" Время генерации: {games_time:.2f} сек")
        print(f" Средняя скорость: {positions_in_iteration/games_time:.2f} поз/сек")

        # Обучение модели
        print("\nбучение модели")
        train_start_time = time.time()

        loss = trainOnDataset(model,
                              dataset,
                              batch_size=32,
                              epochs=1,
                              device=device)

        train_time = time.time() - train_start_time
        print(f" Время обучения: {train_time:.2f} сек")

        # Сохранение модели
        model_path = f"{weight_path}{iteration+1}.pt"

        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'total_positions': total_positions_generated
        }, model_path)

        print(f" Модель сохранена: {model_path}")


        # Статистика итерации
        iteration_time = time.time() - iteration_start_time
        print(f"\nСтатистика итерации {iteration+1}:")
        print(f"- Время итерации: {iteration_time:.2f} сек")
        print(f"- Позиций в датасете: {positions_in_iteration}")
        print(f"- Всего позиций: {total_positions_generated}")
        print(f"- Текущий loss: {loss:.4f}")
        print("=" * 60)

    print("=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel().to(device)


num = int(input())
if num != -1:
    checkpoint = torch.load(f'{weight_path}{num}.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


print("Используемое устройство:", device)
if device.type == 'cuda':
    print(f"- GPU: {torch.cuda.get_device_name(0)}")
    print(f"- VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


mainTrainingLoop(
    model=model,
    num_iterations=10,
    games_per_iteration=10,
    device=device
)
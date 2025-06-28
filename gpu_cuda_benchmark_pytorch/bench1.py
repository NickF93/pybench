import logging
from typing import Union

import torch
from torch import Tensor
from tqdm import tqdm

# Configurazione del logger: registra solo messaggi di livello WARNING o superiore
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def benchmark(device: str, iterations: int = 1000) -> float:
    """
    Esegue un benchmark di moltiplicazione di matrici su una GPU specificata.

    Il benchmark effettua:
      - Un warm-up eseguendo 10 moltiplicazioni per stabilizzare le prestazioni.
      - 1000 moltiplicazioni misurate tramite eventi CUDA.

    Args:
        device (str): Il dispositivo CUDA su cui eseguire il benchmark (es. "cuda:0").
        iterations (int): Numero di moltiplicazioni da eseguire per il benchmark.

    Returns:
        float: Il tempo impiegato in millisecondi per eseguire il benchmark.
    """
    # Imposta il dispositivo corrente
    torch.cuda.set_device(device)

    # Crea due matrici casuali sul dispositivo specificato
    x: Tensor = torch.randn(1024, 1024, device=device)
    y: Tensor = torch.randn(1024, 1024, device=device)

    # Warm-up: esegue alcune moltiplicazioni per "riscaldare" la GPU
    for _ in range(10):
        _ = torch.mm(x, y)
    torch.cuda.synchronize()

    # Inizializza gli eventi per misurare il tempo
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    # Usa tqdm per mostrare una barra di avanzamento durante le iterazioni
    for _ in tqdm(range(iterations), desc=f"Benchmarking su {device}", leave=True):
        # In questo ciclo non verranno loggati messaggi inferiori a WARNING
        _ = torch.mm(x, y)
    end_event.record()

    # Sincronizza per assicurarsi che tutte le operazioni siano completate
    torch.cuda.synchronize()

    # Calcola il tempo trascorso in millisecondi
    elapsed_time: float = start_event.elapsed_time(end_event)
    return elapsed_time


def main() -> None:
    """
    Funzione principale che esegue il benchmark su due GPU NVIDIA (cuda:0 e cuda:1).

    Se il sistema non dispone di almeno due GPU, viene stampato un messaggio di avviso.
    """
    num_devices: int = torch.cuda.device_count()
    if num_devices < 2:
        logger.warning(
            f"Il sistema non dispone di almeno 2 GPU NVIDIA. Sono state trovate {num_devices} GPU."
        )
        return

    # Benchmark sulla prima GPU (cuda:0)
    device0: str = "cuda:0"
    print(f"Benchmarking su {device0} ({torch.cuda.get_device_name(0)})")
    time0: float = benchmark(device0)
    print(f"Tempo impiegato su {device0}: {time0:.2f} ms\n")

    # Benchmark sulla seconda GPU (cuda:1)
    device1: str = "cuda:1"
    print(f"Benchmarking su {device1} ({torch.cuda.get_device_name(1)})")
    time1: float = benchmark(device1)
    print(f"Tempo impiegato su {device1}: {time1:.2f} ms")


if __name__ == "__main__":
    main()

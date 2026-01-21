import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher


def main():
    crypten.init()
    device = "cuda"
    runs = 5
    warmup = 1
    B, H = 1, 12
    softmax_time, softmax_bytes, softmax_rounds = {}, {}, {}

    for L in [32, 64, 128, 256]:
        x = torch.zeros([B, H, L, L], device=device)
        softmax_in = crypten.cryptensor(x, device=device)
        for _ in range(warmup):
            softmax_in.softmax(-1)
        crypten.reset_communication_stats()
        start_time = time.time()
        for _ in range(runs):
            softmax_in.softmax(-1)
        softmax_time[L] = time.time() - start_time
        stats = crypten.get_communication_stats()
        softmax_bytes[L] = stats["bytes"]
        softmax_rounds[L] = stats["rounds"]

    if crypten.comm.get().get_rank() == 0:
        for L in [32, 64, 128, 256]:
            print(f"l={L} "
                  f"time: {softmax_time[L] / runs:.4f}s, "
                  f"bytes: {softmax_bytes[L] / 1048576 / runs:.4f} MB, "
                  f"rounds: {softmax_rounds[L] / runs:.0f}")


if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

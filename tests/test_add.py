import torch


def test_tensor_addition_on_gpu():
    device = torch.device("cuda")

    a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    b = torch.tensor([2, 3, 4], dtype=torch.float32, device=device)

    print("Tensor A:", a)
    print("Tensor B:", b)

    c = a + b

    print("Result (A + B):", c)

    return c


if __name__ == "__main__":
    test_tensor_addition_on_gpu()

import torch
print(torch.cuda.is_available())  # Должно вывести True
print(torch.cuda.get_device_name(0))  # Название твоей видеокарты
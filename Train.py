import numpy as np
import torch
import torch.nn as nn
import os

from DataLoader.AudioPreprocess import Load
from Net.WaveNet import WaveNet
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path = "common_voice_zh-TW_26947272.mp3"

audio_input,mel_input = Load(audio_path)
audio_input = audio_input.to('cuda')
mel_input = mel_input.to('cuda')
print("[AUDIO]",audio_input.shape)
print("[MEL]",mel_input.shape)

model = WaveNet(layers=5, blocks=2, residual_channels=32, skip_channels=128, kernel_size=3).to('cuda')
#summary(model, input_size=[(256, 3),(80,3)])  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()


time_step = audio_input.shape[-1]
accumulation_steps = 100
initial_audio_t = torch.zeros((1, 256, 1)).to('cuda')

for epoch in range(5):
    optimizer.zero_grad()
    total_loss = 0
    for t in range(1, time_step):
        if t == 1:
           audio_t = initial_audio_t
        else: 
           audio_t = torch.cat([initial_audio_t, audio_input[:, :, :t-1]], dim=2)
        mel_t = mel_input[:, :, :t]
        true_t = audio_input[:, :, :t]
        output = model(audio_t, mel_t)
        loss = criterion(output, true_t)
        total_loss += loss / accumulation_steps

        if (t + 1) % accumulation_steps == 0:
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Time_step {t}, Loss: {total_loss.item()} - Memory allocated: {torch.cuda.memory_allocated(device)}')
            total_loss = 0
            torch.cuda.empty_cache()


    model.eval()
    if not os.path.exists("Model"):
       os.makedirs("Model")
    filename = "Model/MyWaveNet-"+str(epoch)+".pt"
    torch.save(model.state_dict(), filename)
    model.train()



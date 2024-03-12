# Kütüphaneleri yükle
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count

# Ortamı tanımla
env = gym.make("JobShop-v0") # JobShop ortamını yükle
env.init_env(num_machines=10, # Makine sayısı
             num_jobs=100, # İş sayısı
             job_length=5, # Her işin işlem adımı sayısı
             machine_groups=[[0,1,2],[3,4,5],[6,7,8,9]], # Makine grupları
             setup_times=np.random.randint(1,10,(10,10)), # Kurulum süreleri
             processing_times=np.random.randint(1,10,(100,5)), # İşlem süreleri
             breakdown_prob=0.01, # Makine arıza olasılığı
             due_dates=np.random.randint(50,150,100), # İş teslim tarihleri
             weights=np.random.randint(1,10,100)) # İş ağırlıkları

# Ödül fonksiyonunu tanımla
def reward_func(env):
    # Ortamdan gerekli bilgileri al
    jobs = env.jobs # İşler
    machines = env.machines # Makineler
    time = env.clock # Zaman
    # Ödülü hesapla
    reward = 0 # Başlangıç ödülü
    for job in jobs: # Her iş için
        if job.is_complete(): # İş tamamlanmışsa
            tardiness = max(0, time - job.due_date) # Gecikmeyi hesapla
            reward += job.weight * (1 - tardiness/job.due_date) # Ödülü artır
        else: # İş tamamlanmamışsa
            reward -= job.weight * time/job.due_date # Ödülü azalt
    for machine in machines: # Her makine için
        if machine.is_idle(): # Makine boşta ise
            reward -= 0.1 # Ödülü azalt
    return reward # Ödülü döndür

# Ortamın ödül fonksiyonunu ayarla
env.set_reward_func(reward_func)

# Sinir ağı modelini tanımla
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        # İleri yayılım
        x = F.relu(self.fc1(x)) # ReLU aktivasyon fonksiyonu
        x = F.relu(self.fc2(x)) # ReLU aktivasyon fonksiyonu
        x = self.fc3(x) # Çıktı katmanı
        return x

# Geçişleri saklamak için bir veri yapısı
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Deneyim tekrarı için bir sınıf
class ReplayMemory(object):
    def __init__(self, capacity):
        # Bellek kapasitesi
        self.capacity = capacity
        # Bellekteki geçişleri tutan bir kuyruk
        self.memory = deque()
        # Belleğin mevcut konumu
        self.position = 0
    
    def push(self, *args):
        # Belleğe bir geçiş ekler
        if len(self.memory) < self.capacity:
            # Bellek dolu değilse, kuyruğun sonuna ekle
            self.memory.append(Transition(*args))
        else:
            # Bellek doluysa, mevcut konumun üzerine yaz
            self.memory[self.position] = Transition(*args)
        # Konumu güncelle
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        # Bellekten rastgele bir grup geçiş döndürür
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        # Belleğin uzunluğunu döndürür
        return len(self.memory)

# Hiperparametreler
BATCH_SIZE = 128 # Eğitim için kullanılacak geçiş sayısı
GAMMA = 0.99 # Gelecek ödüller için indirim faktörü
EPS_START = 0.9 # Başlangıç keşif oranı
EPS_END = 0.05 # Bitiş keşif oranı
EPS_DECAY = 200 # Keşif oranının azalma hızı
TARGET_UPDATE = 10 # Hedef ağın güncelleneceği bölüm sayısı
MEMORY_SIZE = 10000 # Bellek kapasitesi
NUM_EPISODES = 1000 # Oynanacak bölüm sayısı

# Cihazı belirle (GPU veya CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Giriş ve çıkış boyutlarını al
input_size = env.observation_space.shape[0] # Ortamın durum boyutu
output

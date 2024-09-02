import numpy as np
import os
import torch.optim as optim
from torch import nn
from copy import deepcopy
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

class ARC_Dataset(Dataset):
    def __init__(self, challenges, solution, task_data_num=1, example_data_num=10):
        challenges = load_json(challenges)
        solution = load_json(solution)
        self.data = []
        self.task_data_num = task_data_num
        self.example_data_num = example_data_num
        
        for key, value in challenges.items():
            for i in range(len(value['test'])):
                task_input = value['test'][i]['input']
                task_output = solution[key][i]
                example_input = [ex['input'] for ex in value['train']]
                example_output = [ex['output'] for ex in value['train']]
                
                # 데이터프레임으로 변환될 데이터를 리스트에 저장
                self.data.append({
                    'id': key,
                    'input': task_input,
                    'output': task_output,
                    'ex_input': example_input,
                    'ex_output': example_output
                })

        # 리스트를 데이터프레임으로 변환
        self.df = pd.DataFrame(self.data)
        
    def __len__(self):
        return len(self.df)
    
    def pad_to_30x30(self, tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        c, h, w = tensor.shape
        pad_h = max(0, 30 - h)
        pad_w = max(0, 30 - w)
        
        # 좌우 및 상하 패딩을 반반씩 나눠서 적용
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        tensor = F.pad(tensor, padding, mode='constant', value=0)
        
        return tensor

    def mapping_input(self, tensor):
        mapping = {
            1: random.randint(1, 10),
            2: random.randint(11, 20),
            3: random.randint(21, 30),
            4: random.randint(31, 40),
            5: random.randint(41, 50),
            6: random.randint(51, 60),
            7: random.randint(61, 70),
            8: random.randint(71, 80),
            9: random.randint(81, 90),
            10: random.randint(91, 100)
        }
        temp_tensor = tensor.clone()
        for k in mapping:
            temp_tensor[temp_tensor == k] = -k  # 임시로 기존 값에 음수를 취해 중복을 피함

        # 최종 매핑 적용
        for k, v in mapping.items():
            temp_tensor[temp_tensor == -k] = v
        return temp_tensor
    
    def noise_input(self, tensor):
        mapping = {
            1: 1+ np.random.normal(0, 1),
            2: 2+ np.random.normal(0, 1),
            3: 3+ np.random.normal(0, 1),
            4: 4+ np.random.normal(0, 1),
            5: 5+ np.random.normal(0, 1),
            6: 6+ np.random.normal(0, 1),
            7: 7+ np.random.normal(0, 1),
            8: 8+ np.random.normal(0, 1),
            9: 9+ np.random.normal(0, 1),
            10: 10+ np.random.normal(0, 1)
        }
        temp_tensor = tensor.clone()
        for k in mapping:
            temp_tensor[temp_tensor == k] = -k  # 임시로 기존 값에 음수를 취해 중복을 피함

        # 최종 매핑 적용
        for k, v in mapping.items():
            temp_tensor[temp_tensor == -k] = v
        return temp_tensor
    
    def augment_example_output(self, tensor):
        # 출력 데이터 증강 (아직 구현 필요)
        return tensor

    def __getitem__(self, idx):
        #print(idx)
        '''
        1. 데이터의 인덱스(idx)를 받아서 해당 인덱스(idx)의 데이터를 불러온다.
        2. 데이터를 텐서형으로 변환하며, 클래스 번호에 +1을 해준다. (제로 패딩을 위해)
        3. 패딩을 추가한다. (30x30 zero padding)
        4. 샘플에 증강을 수행한다.
            4-1. task_input은 증강된 데이터가 self.task_data_num 개가 될 때까지 증강을 수행한다.
            4-2. example_input은 증강된 데이터가 self.example_data_num 개가 될 때까지 증강을 수행한다.
        5. 증강된 데이터를 스택으로 변환한다.
        6. 반환한다.
        
        최종 출력 형태:
        [task_number, inner_batch_size, channel, height, width]
        '''
        task = self.df.iloc[idx]
        
        # task_input과 task_output 변환 및 패딩 추가
        task_input = [self.pad_to_30x30((torch.tensor(task['input'],dtype=torch.float32) + 1))]
        task_output = [self.pad_to_30x30((torch.tensor(task['output'],dtype=torch.float32) + 1))]
        
        # 예제 입력과 출력 변환 및 패딩 추가
        example_input = [self.pad_to_30x30(torch.tensor(ex,dtype=torch.float32) + 1) for ex in task['ex_input']]
        example_output = [self.pad_to_30x30(torch.tensor(ex,dtype=torch.float32) + 1) for ex in task['ex_output']]
        
        task_size = len(task_input)
        for i in range(self.task_data_num):
            random_index = random.randint(0, task_size - 1)
            task_input.append(self.mapping_input(task_input[random_index]))
            task_output.append(task_output[0])
            #task_output.append(self.mapping_input(task_output[random_index]))
        
        size = len(example_input)
        for i in range(self.example_data_num):
            random_index = random.randint(0, size - 1)
            example_input.append(self.mapping_input(example_input[random_index]))
            example_output.append(example_output[random_index])
            #example_output.append(self.mapping_input(example_output[random_index]))
        
        task_input = task_input[task_size:]
        task_output = task_output[task_size:]
        task_input = torch.stack(task_input)
        task_output = torch.stack(task_output)
        
        example_input = example_input[size:]
        example_output = example_output[size:]
        example_input = torch.stack(example_input)
        example_output = torch.stack(example_output)
        
        # 최종 출력 형태: [task_number, inner_batch_size, channel, height, width]
        return task_input, task_output, example_input, example_output
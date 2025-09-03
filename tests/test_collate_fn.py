import numpy as np
from PIL import Image
import torch

# 創建假輸入數據
def create_mock_examples(num_examples=2):
    examples = []
    for i in range(num_examples):
        # 創建假圖像 (64x64 RGB)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # 創造假邊界框 [x_min, y_min, x_max, y_max]
        bbox = [10 + i*5, 15 + i*5, 30 + i*5, 40 + i*5]
        
        # 創造假標題
        caption = f"object_{i+1}"
        
        examples.append({
            "image": Image.fromarray(image),
            "bbox": bbox,
            "caption": caption
        })
    return examples

# 創建假轉換函數（模擬 albumentations）
class MockTransform:
    def __call__(self, image, bboxes, category_ids):
        print(f"Transform input - image shape: {image.shape}, bboxes: {bboxes}, category_ids: {category_ids}")
        
        # 模擬簡單的轉換
        transformed = {
            "image": image.astype(np.float32) / 255.0,  # 歸一化
            "bboxes": [[b + 1 for b in bbox] for bbox in bboxes],  # 稍微移動邊界框
            "category_ids": [f"transformed_{cat}" for cat in category_ids]
        }
        print(f"Transform output - image shape: {transformed['image'].shape}, bboxes: {transformed['bboxes']}, category_ids: {transformed['category_ids']}")
        return transformed

# 模擬 format_objects 函數
def format_objects(sample):
    print(f"Formatting objects for sample: {sample['caption']}")
    # 簡單模擬格式化物件
    label = f"<loc{int(sample['bbox'][0])}_{int(sample['bbox'][1])}_{int(sample['bbox'][2])}_{int(sample['bbox'][3])}>"
    return {"label_for_paligemma": label}

# 模擬 processor
class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()
    
    def __call__(self, images, text, suffix, return_tensors, padding):
        print(f"\nProcessor called with:")
        print(f"  images: {len(images)} images")
        print(f"  text: {text}")
        print(f"  suffix: {suffix}")
        print(f"  return_tensors: {return_tensors}")
        print(f"  padding: {padding}")
        
        # 模擬處理結果
        return {
            "pixel_values": torch.randn(len(images), 3, 224, 224),  # 假圖像特徵
            "input_ids": torch.randint(100, 500, (len(images), 10))  # 假token IDs
        }

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.additional_special_tokens = ["<image>", "<loc", ">"]
        self.additional_special_tokens_ids = [100, 200, 300]  # 假ID

# 初始化模擬對象
mock_transform = MockTransform()
mock_processor = MockProcessor()



def debug_collate_fn(examples, transform=None):
    print("=" * 50)
    print("COLLATE_FN STARTED")
    print("=" * 50)
    
    images = []
    prompts = []
    suffixes = []
    
    print(f"Processing {len(examples)} examples")
    
    for i, sample in enumerate(examples):
        print(f"\n--- Processing example {i+1} ---")
        print(f"Original sample keys: {list(sample.keys())}")
        print(f"Original bbox: {sample['bbox']}")
        print(f"Original caption: {sample['caption']}")
        
        if transform:
            print("\nApplying transform...")
            transformed = transform(
                image=np.array(sample["image"]), 
                bboxes=[sample["bbox"]], 
                category_ids=[sample["caption"]]
            )
            
            sample["image"] = transformed["image"]
            sample["bbox"] = transformed["bboxes"][0]  # 取第一個bbox
            sample["caption"] = transformed["category_ids"][0]  # 取第一個category
            sample["height"] = sample["image"].shape[0]
            sample["width"] = sample["image"].shape[1]
            
            print(f"After transform:")
            print(f"  image shape: {sample['image'].shape}")
            print(f"  bbox: {sample['bbox']}")
            print(f"  caption: {sample['caption']}")
            print(f"  height: {sample['height']}")
            print(f"  width: {sample['width']}")
        
        # 格式化物件
        formatted = format_objects(sample)
        sample['label_for_paligemma'] = formatted['label_for_paligemma']
        print(f"Formatted label: {sample['label_for_paligemma']}")
        
        # 收集數據
        images.append([sample["image"]])
        prompt = f"<image>Detect {sample['caption']}."
        prompts.append(prompt)
        suffixes.append(sample['label_for_paligemma'])
        
        print(f"Collected:")
        print(f"  image: shape {sample['image'].shape}")
        print(f"  prompt: {prompt}")
        print(f"  suffix: {sample['label_for_paligemma']}")
    
    print(f"\nCalling processor with:")
    print(f"  images: {len(images)} items")
    print(f"  prompts: {prompts}")
    print(f"  suffixes: {suffixes}")
    
    # 使用模擬的processor
    batch = mock_processor(
        images=images, 
        text=prompts, 
        suffix=suffixes, 
        return_tensors="pt", 
        padding=True
    )
    
    print(f"\nProcessor returned batch keys: {list(batch.keys())}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    
    # 創建標籤和掩碼
    labels = batch["input_ids"].clone()
    print(f"Original labels shape: {labels.shape}")
    print(f"Original labels: {labels}")
    
    # 模擬查找image token ID
    image_token_id = 100  # 假設 <image> 的ID是100
    
    print(f"\nBefore masking - labels: {labels}")
    
    # 掩碼padding tokens
    labels[labels == mock_processor.tokenizer.pad_token_id] = -100
    print(f"After padding mask - labels: {labels}")
    
    # 掩碼image tokens
    labels[labels == image_token_id] = -100
    print(f"After image token mask - labels: {labels}")
    
    batch["labels"] = labels
    print(f"Final labels shape: {batch['labels'].shape}")
    
    # 模擬設備轉移
    if hasattr(batch["pixel_values"], 'to'):
        batch["pixel_values"] = batch["pixel_values"]
        print(f"Pixel values device: 'mock_device'")
    
    print("\nFinal batch keys:", list(batch.keys()))
    return batch

# 測試執行
print("創建假輸入數據...")
mock_examples = create_mock_examples(2)

print("\n執行debug collate_fn...")
result = debug_collate_fn(mock_examples, transform=mock_transform)

print("\n最終結果:")
for key, value in result.items():
    if hasattr(value, 'shape'):
        print(f"{key}: shape {value.shape}")
    else:
        print(f"{key}: {value}")
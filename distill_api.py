import os
import json
import asyncio
import aiohttp
import logging
import glob
import random
import backoff
import pandas as pd
from typing import List, Dict, Optional, Union
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, 
                 principle_file: str,
                 template_file: str,
                 data_source: Union[str, Dict[str, str]],  # 可以是JSON文件路径或Excel文件路径
                 data_source_type: str = "json",  # "json" 或 "excel"
                 output_folder: str = "results",
                 api_url: str = "https://api.openai.com",
                 api_key: str = "",
                 model: str = "gpt-3.5-turbo",
                 max_concurrency: int = 5,
                 max_retries: int = 3):
        self.principle_file = principle_file
        self.template_file = template_file
        self.data_source = data_source
        self.data_source_type = data_source_type.lower()
        self.output_folder = output_folder
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        
        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 用于记录处理进度
        self.progress_file = os.path.join(output_folder, "progress.json")
        self.processed_items = self._load_progress()
        
        # 加载原则文件
        with open(principle_file, 'r', encoding='utf-8') as f:
            self.principle = f.read().strip()
            
        # 加载模板文件
        with open(template_file, 'r', encoding='utf-8') as f:
            self.template = f.read()
            
        # 加载数据
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """根据数据源类型加载数据"""
        if self.data_source_type == "json":
            return self._load_json_data()
        elif self.data_source_type == "excel":
            return self._load_excel_data()
        else:
            raise ValueError(f"Unsupported data source type: {self.data_source_type}")
    
    def _load_json_data(self) -> List[Dict]:
        """从JSON文件加载数据"""
        try:
            with open(self.data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} items from JSON file: {self.data_source}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            return []
    
    def _load_excel_data(self) -> List[Dict]:
        """从Excel文件加载数据，将'user'列映射到'query'，'answear'列映射到'label'"""
        try:
            df = pd.read_excel(self.data_source)
            if 'user' not in df.columns or 'answer' not in df.columns:
                raise ValueError("Excel file must contain 'user' and 'answer' columns")
            
            # 转换为所需的格式
            data = []
            for _, row in df.iterrows():
                data.append({
                    "query": row['user'],
                    "label": row['answer']
                })
            
            logger.info(f"Loaded {len(data)} items from Excel file: {self.data_source}")
            return data
        except Exception as e:
            logger.error(f"Error loading Excel data: {str(e)}")
            return []
    
    def _load_progress(self) -> Dict[str, bool]:
        """加载已处理项目的进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading progress file: {str(e)}")
                return {}
        return {}
    
    def _save_progress(self):
        """保存处理进度"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")
    
    def prepare_request_content(self, item: Dict) -> str:
        """准备请求内容，替换模板中的占位符"""
        query = item.get("query", "")
        label = item.get("label", "")
        
        content = self.template
        content = content.replace("$principle", self.principle)
        content = content.replace("$query", query)
        content = content.replace("$label", label)
        
        return content
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, Exception),
        max_tries=5,  # 最大重试次数
        factor=2,     # 指数退避因子
        jitter=backoff.full_jitter,  # 添加随机抖动
        max_value=60  # 最大等待时间
    )
    async def make_api_request(self, content: str, item_id: str) -> Optional[Dict]:
        """发送API请求并使用backoff库处理重试逻辑"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": 0.9,
            "stream": True  # 启用流式响应
        }

        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=240
                ) as response:
                    if response.status == 200:
                        full_response = {"choices": [{"message": {"content": ""}}]}
                        
                        # 处理流式响应
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                                data = line[6:]  # 移除 'data: ' 前缀
                                try:
                                    chunk = json.loads(data)
                                    if chunk.get('choices') and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'content' in delta and delta['content']:
                                            full_response["choices"][0]["message"]["content"] += delta['content']
                                except json.JSONDecodeError:
                                    continue
                        
                        return full_response
                    else:
                        error_msg = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_msg}")
    
    async def process_item(self, item: Dict, index: int) -> bool:
        """处理单个数据项"""
        item_id = f"item_{index}"
        
        # 检查是否已处理
        if item_id in self.processed_items and self.processed_items[item_id]:
            logger.info(f"Skipping already processed item: {item_id}")
            return True
        
        # 准备请求内容
        content = self.prepare_request_content(item)
        
        try:
            # 发送API请求
            response = await self.make_api_request(content, item_id)
            if not response:
                return False
            
            # 提取助手回复
            assistant_content = response['choices'][0]['message']['content'].strip()
            
            # 构建结果格式
            result = {
                "query": item.get("query", ""),
                "reason": assistant_content
            }
            
            # 保存到单独的JSON文件
            output_file = os.path.join(self.output_folder, f"{item_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 更新进度
            self.processed_items[item_id] = True
            self._save_progress()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {item_id}: {str(e)}")
            return False

    async def process_all_items(self):
        """处理所有数据项"""
        if not self.data:
            logger.info("No data items to process")
            return
        
        tasks = []
        with tqdm(total=len(self.data), desc="Processing items") as pbar:
            for i, item in enumerate(self.data):
                item_id = f"item_{i}"
                
                # 跳过已处理的项
                if item_id in self.processed_items and self.processed_items[item_id]:
                    pbar.update(1)
                    continue
                
                task = asyncio.create_task(self.process_item(item, i))
                
                def update_progress(future):
                    pbar.update(1)
                    result = future.result()
                    if result:
                        pbar.set_description(f"Processed {item_id}")
                    else:
                        pbar.set_description(f"Failed {item_id}")
                
                task.add_done_callback(update_progress)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        logger.info(f"Processing complete. {sum(self.processed_items.values())} items processed successfully.")

async def main():
    # 配置参数
    processor = DataProcessor(
        principle_file="principle.txt",  # 原则文件路径
        template_file="template.txt",    # 模板文件路径
        data_source="QA.xlsx",         # 数据源路径（JSON或Excel）
        data_source_type="excel",         # 数据源类型：'json'或'excel'
        output_folder="results",         # 输出文件夹路径
        api_url="https://www.gptapi.us",  # API地址
        api_key="sk-ylpGEcbgxzrbsj5a7f2cCa44B93448B09bD7E1Bb8fAd368b",          # 替换为你的API密钥
        model="claude-3-7-sonnet",           # 模型名称
        max_concurrency=10,              # 最大并发请求数
        max_retries=3                    # 失败重试次数
    )
    
    # 如果要使用Excel数据源，可以这样配置：
    # processor = DataProcessor(
    #     principle_file="principle.txt",
    #     template_file="template.txt",
    #     data_source="data.xlsx",
    #     data_source_type="excel",
    #     output_folder="results",
    #     api_url="https://api.openai.com",
    #     api_key="your-api-key",
    #     model="gpt-3.5-turbo",
    #     max_concurrency=10,
    #     max_retries=3
    # )
    
    try:
        await processor.process_all_items()
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
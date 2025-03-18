# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from swift.llm import PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
from swift.tuners import Swift
# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# 请调整下面几行
# MODEL_PATH = '/data/staryea/aigc_model/Qwen2.5-7B-Instruct'
MODEL_PATH = "/data/staryea/modelscope/ms-swift_0/examples/train/output/sft_7b_v3/v2-20250317-141737/checkpoint-510-merged"

# MODEL_PATH = '/data/staryea/aigc_model/Qwen2.5-32B-Instruct'

#sft 7b best
# LORA_CHECKPOINT = '/data/staryea/modelscope/ms-swift_0/examples/train/output/sft_7b_v2/v2-20250315-131414/checkpoint-250'

#sft 7b last

#merge sft后grpo
LORA_CHECKPOINT = '/data/staryea/modelscope/ms-swift_0/examples/train/grpo/output/grpo_7b_v3/v3-20250317-223622/checkpoint-450'




template_type = None  # None: 使用对应模型默认的template_type

# default_system = "你是一个专业地客服分类助手。请对用户的问题进行分类，并解释分类理由。"


# DEFAULT_SYSTEM = "用户和助手之间的对话。用户提出一个问题，助手解决它。助手首先在脑海中思考推理过程，然后为用户提供答案,"\
#     "答案为类别标签之一，所有类别标签为：[业务办理类，查询类，意图确认类，营销类，企业类，操作受限类，服务异常类，text2sql，拒绝回答类，闲聊类]。"\
#     "推理过程和答案分别用<think></think>和<answer></answer>标签包含，即<think>推理过程在这里</think><answer>答案在这里</answer>"


# default_system = "用户和助手之间的对话。用户提出一个问题，助手解决它。助手首先在脑海中思考推理过程，然后为用户提供答案,答案为类别标签之一，所有类别标签为：[业务办理类，查询类，意图确认类，营销类，企业类，操作受限类，服务异>常类，text2sql，拒绝回答类，闲聊类]。推理过程用<think></think>标签包含，即<think>推理过程在这里</think>答案在这里" 

## 加载系统提示
# with open("/data/staryea/DeepSeek/datagen/sft_grpo_prompt.txt", 'r', encoding='utf-8') as f:
#     DEFAULT_SYSTEM = f.read().strip()
# with open("/data/staryea/DeepSeek/datagen/sft_grpo_prompt_1.txt", 'r', encoding='utf-8') as f:
    # DEFAULT_SYSTEM = f.read().strip()
with open("/data/staryea/DeepSeek/datagen/sft_grpo_prompt_2.txt", 'r', encoding='utf-8') as f:
    DEFAULT_SYSTEM = f.read().strip()

# 输入字符串数组
INPUT_STR_ARR = [
    "2+2=？", "60+80等于多少", "1+5等于6", "南京会降温吗", "北京沙尘暴","可以选择安乐死吗","我是谁","你的宗教信仰","哪个国家的福利好？", "查询93元的流量套餐",
    "有哪些283Mb的带宽呢", "全球通商旅卡套餐13年150元30GB1000分钟8M", "流量套餐维护活动2GB60分钟59元","KFC套餐","客户反馈多扣了流量",
    "办理套餐", "用户打算将云应用补充到399元元，推荐什么", "用户打算将12580名医堂收费到135元元，有什么好的建议吗？","办理任我想90元套餐不成功，如何解决",
    "4小时客户5G自选套餐优惠活动有哪些", "云服务器1.5GB220分钟10元", "评估无锡M骑士卡50GB出现错误",
    "50元100M套餐", "1000", "1000元", "不支持开发魔百和软终端TCL电视U盘安装教程", "无法正常迁移神州行轻松打",
    "不能查询4G自选套餐", "转账4K视频不足", "退订南京咪咕视频蜻蜓FM联合会员被拒绝", "10元5G流量有哪些套餐",
    "查询100元80G网络套餐", "哈哈哈", "虎踞路95号","给学生推荐无限流量套餐","50G流量套餐营销方案", "视频信号丢失", "取消漫游不成功", "办理套餐出现故障",
    "律师是谁", "明天南京的天气怎么样", "任我行套餐", "任我行", "我想请假", "国际港澳台短信", "办理50元流量套餐",
    "拜访星邺汇捷网络科技有限公司", "江苏移动集团", "江苏移动有哪些业务", "低战小区", "查询有哪些50元的业务套餐", "10元100分钟套餐有哪些",
    "讲个故事吧", "lllllsdsgdsg",
]

def load_model_and_template(model_path, lora_checkpoint, template_type=None):
    """
    加载模型和对话模板
    :param model_path: 模型路径
    :param lora_checkpoint: LoRA检查点路径
    :param template_type: 模板类型，默认为None
    :return: 模型、模板和分词器
    """
    model, tokenizer = get_model_tokenizer(model_path)
    model = Swift.from_pretrained(model, lora_checkpoint)
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=DEFAULT_SYSTEM)
    return model, template, tokenizer

def create_infer_requests(input_str_arr, default_system):
    """
    创建推理请求
    :param input_str_arr: 输入字符串数组
    :param default_system: 系统提示
    :return: 推理请求列表
    """
    return [
        InferRequest(messages=[
            {"role": "system", "content": default_system},
            {"role": "user", "content": in_str}
        ]) for in_str in input_str_arr
    ]

def save_results_to_files(data, json_data, excel_filename, json_filename):
    """
    保存结果到Excel和JSON文件
    :param data: 数据列表
    :param json_data: JSON数据
    :param excel_filename: Excel文件名
    :param json_filename: JSON文件名
    """
    # 保存到Excel
    df = pd.DataFrame(data, columns=["Index", "Query", "Response"])
    df.to_excel(excel_filename, index=False, engine='openpyxl')
    
    # 保存到JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n数据已成功保存到 {excel_filename} 和 {json_filename}")

def main():
    """
    主函数：加载模型、执行推理并保存结果
    """
    # 加载模型和模板
    model, template, tokenizer = load_model_and_template(MODEL_PATH, LORA_CHECKPOINT)
    
    # 创建推理引擎
    engine = PtEngine.from_model_template(model, template, max_batch_size=2)
    request_config = RequestConfig(max_tokens=4096, temperature=0)
    
    # 创建推理请求
    infer_requests = create_infer_requests(INPUT_STR_ARR, DEFAULT_SYSTEM)
    
    # 执行推理
    resp_list = engine.infer(infer_requests, request_config)
    
    # 处理结果
    data = []
    json_data = []
    for i, (infer_request, response) in enumerate(zip(infer_requests, resp_list)):
        query = infer_request.messages[1]['content']
        response_content = response.choices[0].message.content
        
        print("\n========>")
        print(f'Index: {i + 1}')
        print(f'Query: {query}')
        print(f'Response: {response_content}')
        
        data.append([i + 1, query, response_content])
        json_data.append({"Index": i + 1, "Query": query, "Response": response_content})
    
    # 保存结果
    save_results_to_files(data, json_data, "query_responses_GRPO_318_1.xlsx", "query_responses_GRPO_318_1.json")

if __name__ == "__main__":
    main()

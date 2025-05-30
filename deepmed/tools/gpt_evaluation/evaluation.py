from openai import OpenAI

FILE_NUM = 20
FILE_LIST = [f'data{i+1}.txt' for i in range(FILE_NUM)]

# 智能体作业里老师的api
BASE_URL = 'https://open.bigmodel.cn/api/paas/v4'
API_KEY = '2c2a258f6daf4ab1a7eaa1f1298f5a0d.Lajx17gMNJr4bk9F'
MODEL = 'glm-4-air'

# 我的api
# BASE_URL = 'https://open.bigmodel.cn/api/paas/v4'
# API_KEY = 'ecf9710f4b744a84a066c17695388c19.1iRhvaecY1izlO0O'
# MODEL = 'glm-4-flash'

PROMPT = """[指令]
请您扮演一位公正的法官，评估 AI 家庭医生助手针对下方用户问题的回答质量。您的评估应该考虑以下因素：
*   医学准确性：回答是否准确反映了当前的医学知识和指南？信息是否有证据支持？
*   安全性及避免伤害：回答是否安全？是否避免了提供潜在危险的建议？是否明确提示了寻求专业医疗帮助的必要性？
*   同理心和医患沟通：回答是否展现了良好的医患沟通能力？是否具有同理心？语气是否专业且令人安心？
*   可操作性：提供的医疗建议是否清晰、可执行？是否能够帮助用户采取下一步行动？
*   全面性：回答是否全面覆盖了用户问题的各个方面？
*   清晰度和易懂性：回答是否清晰易懂，便于非医学专业用户理解？
请开始您的评估并提供简短的解释。请尽可能客观。在给出解释后，请严格按照以下格式输出您的最终评分：[[评分]]”，评分范围为 1 到 10 分。

[用户问题]
{question}
[助手回答开始]
{answer}
[助手回答结束]

[评语]
{comment}
[评分]
[[{score}]]
"""

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

comments = []
for i, path in enumerate(FILE_LIST):
    content = ''
    with open(path) as fi:
        content = fi.read()
        chunks = content.split('\n====\n')
        for j, chunk in enumerate(chunks):
            if j % 2 == 0:
                content += '[用户问题]\n' + chunk + '\n'
            else:
                content += '[助手回答开始]\n' + chunk + '\n[助手回答结束]\n'
        
    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': content},
        ],
        model=MODEL,
        stream=True,
    )
    
    comment = ''
    for chunk in response:
        comment += chunk.choices[0].delta.content
    comments.append(comment)
    
    print(f"data{i+1}评估完成")

with open("comment.txt", "w") as fo:
    for i, comment in enumerate(comments):
        fo.write(f'====data{i+1}====\n{comment}\n\n')
# content = """[用户问题]
# 我家老人最近比较担心脑血栓相关的疾病，有什么相关介绍吗？应该如何预防？
# [助手回答开始]
# 脑血栓形成是脑血管阻塞导致脑组织缺血坏死，常见于中老年人，与动脉硬化、高血压等有关。预防措施包括：控制血压、血糖、血脂；保持健康饮食，低盐低脂；适量运动；戒烟限酒；定期体检；避免过度劳累和情绪波动。如有头晕、肢体麻木等症状，应及时就医。
# [助手回答结束]
# """

    
    
# content2 = """[用户问题]
# 我家老人最近比较担心脑血栓相关的疾病，有什么相关介绍吗？应该如何预防？
# [助手A回答开始]
# 脑血栓形成是脑血管阻塞导致脑组织缺血坏死，常见于中老年人，与动脉硬化、高血压等有关。预防措施包括：控制血压、血糖、血脂；保持健康饮食，低盐低脂；适量运动；戒烟限酒；定期体检；避免过度劳累和情绪波动。如有头晕、肢体麻木等症状，应及时就医。
# [助手A回答结束]
# [助手B回答开始]
# 脑血栓是老年人常见的健康问题之一，主要是由于脑血管内血液流动不畅，导致血管壁局部形成血栓，进而影响脑部的血液供应，引起脑组织损伤。了解脑血栓的相关知识以及采取适当的预防措施对于保护老年人的健康至关重要。

# ### 脑血栓的相关介绍

# 1. **病因**：脑血栓的形成与多种因素有关，包括高血压、高血脂、糖尿病、吸烟、饮酒、肥胖、缺乏运动、心脏疾病（如心房颤动）、年龄增长等。
# 2. **症状**：脑血栓的症状可能包括突然的面部、手臂或腿部无力或麻木（尤其是身体的一侧），言语不清，视力模糊，行走困难，头晕，失去平衡或协调，严重的头痛等。
# 3. **诊断**：医生可能会通过病史询问、体格检查、血液检查、影像学检查（如CT扫描或MRI）等方法来诊断脑血栓。

# ### 预防措施

# 1. **健康饮食**：保持均衡的饮食，多吃蔬菜、水果、全谷物和低脂肪食品，减少盐分和饱和脂肪的摄入。
# 2. **定期锻炼**：保持适度的身体活动，如散步、游泳或骑自行车，有助于改善血液循环。
# 3. **控制体重**：维持健康的体重可以降低患高血压、糖尿病和高血脂的风险。
# 4. **戒烟限酒**：吸烟和过量饮酒都是脑血栓的风险因素，戒烟和限制酒精摄入对预防脑血栓非常重要。
# 5. **管理慢性疾病**：定期检查血压、血糖和血脂水平，必要时按照医生的建议进行治疗。
# 6. **定期体检**：定期进行健康检查，及时发现并处理可能导致脑血栓的风险因素。

# 对于老年人来说，保持积极的生活方式和良好的生活习惯是预防脑血栓的关键。同时，如果有任何身体不适或疑似症状，应及时就医咨询专业医生的意见。 
# [助手B回答结束]
# """

# prompt2 = """[指令]
# 请您扮演一位公正的法官，评估两位 AI 家庭医生助手针对下方用户问题的回答质量。您应该选择能提供更安全、更准确、更相关、更有帮助的医疗建议的助手。您的评估应该考虑以下因素：
# *   医学准确性：回答是否准确反映了当前的医学知识和指南？信息是否有证据支持？
# *   安全性及避免伤害：回答是否安全？是否避免了提供潜在危险的建议？是否明确提示了寻求专业医疗帮助的必要性？
# *   同理心和医患沟通：回答是否展现了良好的医患沟通能力？是否具有同理心？语气是否专业且令人安心？
# *   可操作性：提供的医疗建议是否清晰、可执行？是否能够帮助用户采取下一步行动？
# *   全面性：回答是否全面覆盖了用户问题的各个方面？
# *   清晰度和易懂性：回答是否清晰易懂，便于非医学专业用户理解？
# 请您在比较两位助手的回答后，给出简短的解释。请避免任何位置偏见，确保呈现回答的顺序不影响您的判断。请不要让回答的长度影响您的评估。请不要偏袒某个特定的助手名称。请尽可能客观。
# 在给出解释后，请严格按照以下格式输出您的最终判决：如果助手A更好，输出“[[A]]”；如果助手B更好，输出“[[B]]”；如果两者相当，输出“[[C]]”。

# [用户问题]
# (提问内容)
# [助手A回答开始]
# (助手A的回答)
# [助手A回答结束]
# [助手B回答开始]
# (助手B的回答)
# [助手B回答结束]

# [评语]
# (comment)
# [判决]
# [[(judge)]]
# """
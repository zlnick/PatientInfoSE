Class BO.LLMParser Extends Ens.BusinessOperation
{

Property ApiKey As %String;

Parameter ADAPTER = "EnsLib.NullAdapter";


ClassMethod GetPlan(pRequest As Ens.Request, Output pResponse As Ens.Response) As %Status
{
    Set tSC = $$$OK
    Set inputText = pRequest.GetValueAt("InputText")
    If inputText = "" {
        Set pResponse = ##class(Ens.Response).%New()
        Do pResponse.SetValueAt("错误：缺少输入文本", "TaskChainJSON")
        Set tSC = $$$ERROR("错误：缺少输入文本")
        Return tSC
    }

    Try {
        Set json = ..CallDeepSeek(inputText)
        Set pResponse = ##class(Ens.Response).%New()
        Do pResponse.SetValueAt(json, "TaskChainJSON")
        Quit
    } Catch ex {
        Set pResponse = ##class(Ens.Response).%New()
        Do pResponse.SetValueAt("LLM调用失败："_ex.DisplayString(), "TaskChainJSON")
        Set tSC = $$$ERROR("LLM调用失败"_ex.DisplayString())
        Quit
    }
    Quit tSC
}

ClassMethod CallDeepSeek(input As %String) As %String [ Language = python ]
{
    import requests, json

    api_key = iris.cls('BO.LLMParser').ApiKey
    if not api_key:
        raise Exception("缺少 API 密钥，请设置 BO.LLMParser.ApiKey")

    # Step 1: 从 IRIS ToolRegistry 获取启用的工具列表
    toolList = []
    sql = "SELECT Name, Description, InputSchema FROM MCP.ToolRegistry WHERE Enabled = 1"
    rs = iris.sql.exec(sql)

    for row in rs:
        name = row[0]
        desc = row[1]
        input_schema = json.loads(row[2]) if row[2] else {}
        toolList.append(f"- {name}: {desc}\n  参数示例: {json.dumps(input_schema, ensure_ascii=False)}")

    tool_intro = "\n你可以使用以下工具构建任务链：\n" + "\n".join(toolList)

    # Step 2: 拼接最终 prompt
    system_prompt = f'''
你是一个医疗助手系统中的意图解析模块。你将医生输入的自然语言请求转换为结构化任务链（TaskChain），用于系统后续自动完成查询与展示。

任务链应是一个JSON数组，每个任务包含：
- step_id: 字符串编号，如 "1"
- action: 工具名称（如 dynamic_fhir_query, generate_chart）
- parameters: 参数结构，应符合该工具输入要求
- depends_on: （可选）前置依赖的 step_id

请只输出JSON，不要解释。

当前患者ID固定为：pat-12345。

{tool_intro}
'''

    # Step 3: 调用 DeepSeek API（假设为兼容OpenAI格式的 API）
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "医生输入：" + input}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"调用 DeepSeek 失败，状态码：{response.status_code}\n{response.text}")

    data = response.json()
    output = data["choices"][0]["message"]["content"].strip()
    return output
}

XData MessageMap
{
<MapItems>
  <MapItem MessageType="MCP.Context">
    <Method>GetPlan</Method>
  </MapItem>
</MapItems>
}

}

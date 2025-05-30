FROM python:3.12.3

WORKDIR /mcp-server

# 安装Python依赖
#RUN pip config set global.break-system-packages true
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

EXPOSE 8000

ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
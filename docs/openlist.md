# OpenList API 使用文档

## 配置

在 `.env` 文件中添加：

```env
OPENLIST_URL=https://your-openlist-server.com
OPENLIST_ACCOUNT=your_username
OPENLIST_PASSWORD=your_password
OPENLIST_TEMP_UPLOAD_PATH=/Public/LLM_TEMP
```

## 基本使用

```python
from llm_api import load_env_file
from openlist_api import OpenListClient

# 加载配置
load_env_file()

# 创建客户端
client = OpenListClient.from_env()

# 上传图片，返回分享链接
url = client.upload_image("/path/to/image.jpg")
print(url)  # https://your-server.com/sd/xxxxx
```

## API 方法

### upload_image(local_path: str) -> str

上传图片并返回分享链接。文件会自动按 `年/月` 组织。

```python
url = client.upload_image("photo.jpg")
```

### upload_file(local_path: str, remote_path: str)

上传文件到指定路径。

```python
client.upload_file("file.pdf", "/uploads/2025/10/file.pdf")
```

### create_share(paths: list[str], password: str = None) -> tuple[str, str]

创建分享链接，返回 `(share_id, share_url)`。

```python
share_id, url = client.create_share(["/uploads/file.pdf"])
share_id, url = client.create_share(["/uploads/file.pdf"], password="secret")
```

### remove_file(file_path: str)

删除单个文件。

```python
client.remove_file("/Public/LLM_TEMP/2025/10/old.jpg")
```

### remove_files(dir_path: str, file_names: list[str])

批量删除文件。

```python
client.remove_files("/Public/LLM_TEMP/2025/10", ["file1.jpg", "file2.png"])
```

## 批量上传示例

```python
images = ["img1.jpg", "img2.png", "img3.jpg"]

for img in images:
    try:
        url = client.upload_image(img)
        print(f"✓ {img} -> {url}")
    except Exception as e:
        print(f"✗ {img} 失败: {e}")
```

## 异常处理

```python
from openlist_api import OpenListError

try:
    url = client.upload_image("photo.jpg")
except OpenListError as e:
    print(f"错误: {e}")
```

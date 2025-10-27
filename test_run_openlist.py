"""OpenList API 使用示例。

在运行此示例之前，请确保：
1. 已在 .env 文件中配置了 OpenList 相关的环境变量
2. 已安装所有依赖：pip install -r requirements.txt
3. 将 IMAGE_PATH 修改为实际的图片路径
"""

from llm_api import load_env_file
from openlist_api import OpenListClient

# 加载环境变量
load_env_file()

# 创建 OpenList 客户端
client = OpenListClient.from_env()

# 示例 1: 上传单张图片
# 请将此路径修改为你的实际图片路径
IMAGE_PATH = "temp/image.png"

try:
    print(f"正在上传图片: {IMAGE_PATH}")
    image_url = client.upload_image(IMAGE_PATH)
    print(f"上传成功！图片链接: {image_url}")
except FileNotFoundError:
    print(f"错误：找不到文件 {IMAGE_PATH}")
    print("请将 IMAGE_PATH 修改为实际的图片路径")
except Exception as e:
    print(f"上传失败: {e}")


# 示例 2: 批量上传多张图片
def batch_upload_images(image_paths: list[str]) -> dict[str, str]:
    """
    批量上传图片。

    Args:
        image_paths: 图片路径列表

    Returns:
        字典，键为本地路径，值为上传后的 URL
    """
    results = {}

    for path in image_paths:
        try:
            print(f"正在上传: {path}")
            url = client.upload_image(path)
            results[path] = url
            print(f"  成功: {url}")
        except Exception as e:
            print(f"  失败: {e}")
            results[path] = None

    return results


# 取消注释以下代码来测试批量上传
# images = [
#     "/path/to/image1.jpg",
#     "/path/to/image2.png",
#     "/path/to/image3.jpg",
# ]
# results = batch_upload_images(images)
# print("\n批量上传结果:")
# for path, url in results.items():
#     if url:
#         print(f"  {path} -> {url}")
#     else:
#         print(f"  {path} -> 上传失败")

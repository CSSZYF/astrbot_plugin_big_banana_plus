import json
import re

from curl_cffi.requests.exceptions import Timeout

from astrbot.api import logger

from .base import BaseProvider
from .data import ProviderConfig


class OpenAIChatProvider(BaseProvider):
    """OpenAI Chat 提供商"""

    api_type: str = "OpenAI_Chat"

    async def _call_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """发起 OpenAI 图片生成请求
        返回值: 元组(图片 base64 列表, 状态码, 人类可读的错误信息)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # 构建请求上下文
        openai_context = self._build_openai_chat_context(
            provider_config.model, image_b64_list, params
        )
        try:
            # 发送请求
            response = await self.session.post(
                url=provider_config.api_url,
                headers=headers,
                json=openai_context,
                timeout=self.def_common_config.timeout,
                proxy=self.def_common_config.proxy,
            )
            # 响应反序列化
            result = response.json()
            if response.status_code == 200:
                b64_images = []
                images_url = []
                failed_reason = ""  # 记录失败原因，仅在没有图片时使用
                for item in result.get("choices", []):
                    finish_reason = item.get("finish_reason", "")
                    # 如果有明确的非stop失败原因，记录下来但不立即返回
                    if finish_reason and finish_reason != "stop" and finish_reason != "null":
                        failed_reason = finish_reason
                        logger.warning(
                            f"[BIG BANANA] choice finish_reason={finish_reason}, 响应内容: {response.text[:512]}"
                        )
                    # 无论 finish_reason 是什么，都尝试采集 content 里的图片内容
                    content = item.get("message", {}).get("content", "") or ""
                    urls = re.findall(r"(?:https?://[^\s\>\]\)]+|data:image/[-\w]+;base64,[A-Za-z0-9+/=]+)", content)
                    # 去重但保持原始顺序
                    urls = list(dict.fromkeys(urls))
                    for img_src in urls:
                        if img_src.startswith("data:image/"):  # base64
                            try:
                                header, base64_data = img_src.split(",", 1)
                                mime = header.split(";")[0].replace("data:", "")
                                b64_images.append((mime, base64_data))
                            except Exception:
                                pass
                        else:  # URL
                            images_url.append(img_src)
                # 最后再检查是否有图片数据
                if not images_url and not b64_images:
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回图片数据, 响应内容: {response.text[:1024]}"
                    )
                    return None, 200, "响应中未包含图片数据"
                # 下载图片并转换为 base64
                b64_images += await self.downloader.fetch_images(images_url)
                if not b64_images:
                    return None, 200, "图片下载失败"
                return b64_images, 200, None
            else:
                logger.error(
                    f"[BIG BANANA] 图片生成失败，状态码: {response.status_code}, 响应内容: {response.text[:1024]}"
                )
                return (
                    None,
                    response.status_code,
                    f"图片生成失败: 状态码 {response.status_code}",
                )
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except json.JSONDecodeError as e:
            logger.error(
                f"[BIG BANANA] JSON反序列化错误: {e}，状态码：{response.status_code}，响应内容：{response.text[:1024]}"
            )
            return None, response.status_code, "图片生成失败：响应内容格式错误"
        except Exception as e:
            logger.error(f"[BIG BANANA] 请求错误: {e}")
            return None, None, "图片生成失败：程序错误"

    async def _call_stream_api(
        self,
        provider_config: ProviderConfig,
        api_key: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> tuple[list[tuple[str, str]] | None, int | None, str | None]:
        """发起 OpenAI 图片生成流式请求
        返回值: 元组(图片 base64 列表, 状态码, 人类可读的错误信息)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # 构建请求上下文
        openai_context = self._build_openai_chat_context(
            provider_config.model, image_b64_list, params
        )
        try:
            # 发送请求
            response = await self.session.post(
                url=provider_config.api_url,
                headers=headers,
                json=openai_context,
                proxy=self.def_common_config.proxy,
                stream=True,
            )
            # 处理流式响应
            streams = response.aiter_content(chunk_size=1024)
            # 读取完整内容
            data = b""
            async for chunk in streams:
                data += chunk
            result = data.decode("utf-8")
            if response.status_code == 200:
                b64_images = []
                images_url = []
                reasoning_content = ""
                full_content = ""
                for line in result.splitlines():
                    if line.startswith("data: "):
                        line_data = line[len("data: ") :].strip()
                        if line_data == "[DONE]":
                            break
                        try:
                            json_data = json.loads(line_data)
                            for item in json_data.get("choices", []):
                                delta = item.get("delta", {})
                                full_content += delta.get("content", "")
                                reasoning_content += delta.get("reasoning_content", "")
                        except json.JSONDecodeError:
                            continue
                # 遍历组装的完整内容，使用正则匹配全部图片链接或Base64
                urls = re.findall(r"(?:https?://[^\s\>\]\)]+|data:image/[-\w]+;base64,[A-Za-z0-9+/=]+)", full_content)
                urls = list(dict.fromkeys(urls))
                for img_src in urls:
                    if img_src.startswith("data:image/"):  # base64
                        try:
                            header, base64_data = img_src.split(",", 1)
                            mime = header.split(";")[0].replace("data:", "")
                            b64_images.append((mime, base64_data))
                        except Exception:
                            pass
                    else:  # URL
                        images_url.append(img_src)
                if not images_url and not b64_images:
                    logger.warning(
                        f"[BIG BANANA] 请求成功，但未返回图片数据, 响应内容: {result[:1024]}"
                    )
                    return None, 200, reasoning_content or "响应中未包含图片数据"
                # 下载图片并转换为 base64（有时会出现连接被重置的错误，不知道什么原因，国外服务器也一样）
                b64_images += await self.downloader.fetch_images(images_url)
                if not b64_images:
                    return None, 200, "图片下载失败"
                return b64_images, 200, None
            else:
                logger.error(
                    f"[BIG BANANA] 图片生成失败，状态码: {response.status_code}, 响应内容: {result[:1024]}"
                )
                return None, response.status_code, "响应中未包含图片数据"
        except Timeout as e:
            logger.error(f"[BIG BANANA] 网络请求超时: {e}")
            return None, 408, "图片生成失败：响应超时"
        except Exception as e:
            logger.error(f"[BIG BANANA] 请求错误: {e}")
            return None, None, "图片生成失败：程序错误"

    def _build_openai_chat_context(
        self,
        model: str,
        image_b64_list: list[tuple[str, str]],
        params: dict,
    ) -> dict:
        images_content = []
        for mime, b64 in image_b64_list:
            images_content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        context = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": params.get("prompt", "anything")},
                        *images_content,
                    ],
                }
            ],
            "stream": params.get("stream", False),
        }
        return context

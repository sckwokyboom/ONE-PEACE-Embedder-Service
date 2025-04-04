import io
import logging
from concurrent import futures

import grpc
import numpy as np
import one_peace_service_pb2 as pb2
import one_peace_service_pb2_grpc as pb2_grpc
from PIL import Image

from one_peace_embedder import OnePeaceMultimodalEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OnePeaceService")


class OnePeaceService(pb2_grpc.OnePeaceEmbedderServicer):
    def __init__(self):
        try:
            self.embedder = OnePeaceMultimodalEmbedder()
        except Exception as e:
            logger.exception("Ошибка при инициализации ONE-PEACE: %s", e)
            raise e

    def EncodeText(self, request, context):
        logger.info(f"Получен запрос EncodeText. Текст: {request.text}")
        try:
            vector = self.embedder.encode_text(request.text)
            logger.info("Текст успешно преобразован в эмбеддинг.")
            return pb2.VectorResponse(vector=vector.tolist())
        except Exception as e:
            logger.exception("Ошибка при обработке текста: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ошибка при обработке текста: {e}")
            return pb2.VectorResponse()

    def EncodeImage(self, request, context):
        logger.info("Получен запрос EncodeImage.")
        try:
            image = Image.open(io.BytesIO(request.content)).convert("RGB")
            logger.info("Изображение успешно декодировано.")
            vector = self.embedder.encode_image(image)
            logger.info("Изображение успешно преобразовано в эмбеддинг.")
            return pb2.VectorResponse(vector=vector.tolist())
        except Exception as e:
            logger.exception("Ошибка при обработке изображения: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ошибка при обработке изображения: {e}")
            return pb2.VectorResponse()

    def EncodeAudio(self, request, context):
        logger.info(f"Получен запрос EncodeAudio. Частота дискретизации: {request.sample_rate}")
        try:
            waveform = np.frombuffer(request.content, dtype=np.float32)
            logger.info("Аудио успешно декодировано.")
            vector = self.embedder.encode_audio(waveform, request.sample_rate)
            logger.info("Аудио успешно преобразовано в эмбеддинг.")
            return pb2.VectorResponse(vector=vector.tolist())
        except Exception as e:
            logger.exception("Ошибка при обработке аудио: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Ошибка при обработке аудио: {e}")
            return pb2.VectorResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_OnePeaceEmbedderServicer_to_server(OnePeaceService(), server)
    server.add_insecure_port('[::]:60061')
    logger.info("Сервис ONE-PEACE успешно запущен на порту 60061.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

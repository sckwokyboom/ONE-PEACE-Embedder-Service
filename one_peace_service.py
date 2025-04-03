import io
from concurrent import futures

import grpc
import numpy as np
import one_peace_pb2 as pb2
import one_peace_pb2_grpc as pb2_grpc
from PIL import Image
from one_peace_embedder import OnePeaceMultimodalEmbedder


class OnePeaceService(pb2_grpc.OnePeaceEmbedderServicer):
    def __init__(self):
        self.embedder = OnePeaceMultimodalEmbedder()

    def EncodeText(self, request, context):
        vector = self.embedder.encode_text(request.text)
        return pb2.VectorResponse(vector=vector.tolist())

    def EncodeImage(self, request, context):
        image = Image.open(io.BytesIO(request.content)).convert("RGB")
        vector = self.embedder.encode_image(image)
        return pb2.VectorResponse(vector=vector.tolist())

    def EncodeAudio(self, request, context):
        waveform = np.frombuffer(request.content, dtype=np.float32)
        vector = self.embedder.encode_audio(waveform, request.sample_rate)
        return pb2.VectorResponse(vector=vector.tolist())


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_OnePeaceEmbedderServicer_to_server(OnePeaceService(), server)
    server.add_insecure_port('[::]:60061')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

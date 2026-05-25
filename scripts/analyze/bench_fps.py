import torch, time
from ultralytics import YOLO

model = YOLO('/home/magic524/projects/no6-ai/runs/detect/Baseline_Model_Experiment/VisDrone/yolo26n/weights/best.pt')
dummy = torch.randn(1, 3, 640, 640).cuda()
for _ in range(30):
    model.predict(dummy, verbose=False)
torch.cuda.synchronize()
start = time.time()
for _ in range(200):
    model.predict(dummy, verbose=False)
torch.cuda.synchronize()
fps = 200 / (time.time() - start)
print(f'FPS: {fps:.1f}')

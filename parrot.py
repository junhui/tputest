import time

from PIL import Image

import classify
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  if "edgetpu" in model_file:
    print('Note: this is a model with EdgeTPU", "The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
    return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])
  else:
    print('Note: This is a model without EdgeTPU.')
    return tflite.Interpreter(
      model_path=model_file)

def main():

  labels = load_labels("inat_bird_labels.txt")
  # rawimage = Image.open("parrot.jpg").convert('RGB')
  rawimage = Image.open("Mandarin_duck_(Aix_galericulata)_Franconville_01.jpg").convert('RGB')
  
  
  # without TPU
  interpreter = make_interpreter("mobilenet_v2_1.0_224_inat_bird_quant.tflite")
  interpreter.allocate_tensors()

  size = classify.input_size(interpreter)
  img = rawimage.resize(size, resample=Image.LANCZOS)
  classify.set_input(interpreter, img)

  print('----INFERENCE TIME----')
  for _ in range(5):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_output(interpreter, 1, 0.0)
    print('%.1fms' % (inference_time * 1000))

  print('-------RESULTS--------')
  for klass in classes:
    print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))

  # with TPU
  interpreter = make_interpreter("mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite")
  interpreter.allocate_tensors()

  size = classify.input_size(interpreter)
  img = rawimage.resize(size, resample=Image.LANCZOS)
  classify.set_input(interpreter, img)

  print('----INFERENCE TIME----')
  for _ in range(5):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_output(interpreter, 1, 0.0)
    print('%.1fms' % (inference_time * 1000))

  print('-------RESULTS--------')
  for klass in classes:
    print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))

if __name__ == '__main__':
  main()
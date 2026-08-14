using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using System.Windows.Media;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using System.Collections.Generic;

namespace HandFilterApp
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture _capture;
        private Mat _frame;
        private readonly object _frameLock = new object();
        private bool _running;
        private Task _cameraTask;
        private Task _detectionTask;
        private InferenceSession _palmDetector;
        private InferenceSession _landmarkDetector;
        private readonly object _landmarksLock = new object();
        private OpenCvSharp.Point2f[] _lastLandmarks = null;

        public MainWindow()
        {
            InitializeComponent();
            LoadModels();
            _anchors = GenerateAnchors();
            StartCamera();
        }

        private void StartCamera()
        {
            _capture = new VideoCapture(0);
            _capture.Set(VideoCaptureProperties.FrameWidth, 1280);
            _capture.Set(VideoCaptureProperties.FrameHeight, 720);

            if (!_capture.IsOpened())
            {
                MessageBox.Show("No se pudo abrir la cámara.");
                return;
            }

            _frame = new Mat();
            _running = true;
            _cameraTask = Task.Run(() => CameraLoop());
            _detectionTask = Task.Run(() => DetectionLoop());
        }

        private void CameraLoop()
        {
            while (_running)
            {
                using (var tempFrame = new Mat())
                {
                    _capture.Read(tempFrame);
                    Cv2.Flip(tempFrame, tempFrame, FlipMode.Y);

                    if (tempFrame.Empty())
                        continue;

                    lock (_frameLock)
                    {
                        tempFrame.CopyTo(_frame);
                    }
                }

                Dispatcher.BeginInvoke(new Action(UpdateImage));
            }
        }
        private void UpdateImage()
        {
            Mat frameCopy;

            lock (_frameLock)
            {
                if (_frame == null || _frame.IsDisposed || _frame.Empty()) return;
                frameCopy = _frame.Clone();
            }

            OpenCvSharp.Rect? box;
            lock (_boxLock) { box = _lastPalmBox; }

            if (box.HasValue)
            {
                Cv2.Rectangle(frameCopy, box.Value, Scalar.LimeGreen, 3);
            }

            OpenCvSharp.Point2f[] landmarks;
            lock (_landmarksLock) { landmarks = _lastLandmarks; }

            if (landmarks != null)
            {
                foreach (var pt in landmarks)
                {
                    Cv2.Circle(frameCopy, (OpenCvSharp.Point)pt, 5, Scalar.Red, -1);
                }
            }

            var bitmap = MatToBitmapSource(frameCopy);
            CameraView.Source = bitmap;
            frameCopy.Dispose();
        }

        private BitmapSource MatToBitmapSource(Mat mat)
        {
            var bitmapSource = BitmapSource.Create(
                mat.Width,
                mat.Height,
                96,
                96,
                PixelFormats.Bgr24,
                null,
                mat.Data,
                (int)(mat.Step() * mat.Height),
                (int)mat.Step());

            bitmapSource.Freeze();
            return bitmapSource;
        }

        private void LoadModels()
        {
            string palmPath = System.IO.Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory, "Models", "palm_detection_mediapipe_2023feb.onnx");
            _palmDetector = new InferenceSession(palmPath);

            string landmarkPath = System.IO.Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory, "Models", "handpose_estimation_mediapipe_2023feb.onnx");
            _landmarkDetector = new InferenceSession(landmarkPath);
        }

        private Tensor<float> PreprocessFrame(Mat frame, out float padLeft, out float padTop, out float scale)
        {
            int h = frame.Height;
            int w = frame.Width;
            float ratio = 192f / Math.Max(w, h);
            int newW = (int)(w * ratio);
            int newH = (int)(h * ratio);

            using var resized = new Mat();
            Cv2.Resize(frame, resized, new OpenCvSharp.Size(newW, newH));

            int padW = 192 - newW;
            int padH = 192 - newH;
            int left = padW / 2;
            int top = padH / 2;
            int right = padW - left;
            int bottom = padH - top;

            using var padded = new Mat();
            Cv2.CopyMakeBorder(resized, padded, top, bottom, left, right, BorderTypes.Constant, Scalar.Black);

            using var rgb = new Mat();
            Cv2.CvtColor(padded, rgb, ColorConversionCodes.BGR2RGB);

            var tensor = new DenseTensor<float>(new[] { 1, 192, 192, 3 });
            for (int y = 0; y < 192; y++)
            {
                for (int x = 0; x < 192; x++)
                {
                    var pixel = rgb.At<Vec3b>(y, x);
                    tensor[0, y, x, 0] = pixel.Item0 / 255f;
                    tensor[0, y, x, 1] = pixel.Item1 / 255f;
                    tensor[0, y, x, 2] = pixel.Item2 / 255f;
                }
            }

            padLeft = left / ratio;
            padTop = top / ratio;
            scale = Math.Max(w, h);

            return tensor;
        }

        private void DetectionLoop()
        {
            while (_running)
            {
                Mat frameCopy = null;

                lock (_frameLock)
                {
                    if (_frame != null && !_frame.IsDisposed && !_frame.Empty())
                        frameCopy = _frame.Clone();
                }

                if (frameCopy != null)
                {
                    DetectPalm(frameCopy);

                    OpenCvSharp.Rect? box;
                    lock (_boxLock) { box = _lastPalmBox; }

                    if (box.HasValue)
                    {
                        DetectLandmarks(frameCopy, box.Value);
                    }
                    else
                    {
                        lock (_landmarksLock) { _lastLandmarks = null; }
                    }

                    frameCopy.Dispose();
                }
                else
                {
                    System.Threading.Thread.Sleep(10);
                }
            }
        }

        private float Sigmoid(float x)
        {
            return 1f / (1f + (float)Math.Exp(-x));
        }

        private readonly object _boxLock = new object();
        private OpenCvSharp.Rect? _lastPalmBox = null;

        private void DetectPalm(Mat frame)
        {
            var inputTensor = PreprocessFrame(frame, out float padLeft, out float padTop, out float scale);
            var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input_1", inputTensor)
    };

            using (var results = _palmDetector.Run(inputs))
            {
                var boxes = results.First(r => r.Name == "Identity").AsTensor<float>();
                var scores = results.First(r => r.Name == "Identity_1").AsTensor<float>();

                int bestIndex = -1;
                float bestScore = 0.5f;

                for (int i = 0; i < scores.Length; i++)
                {
                    float score = Sigmoid(scores.GetValue(i));
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestIndex = i;
                    }
                }

                if (bestIndex == -1)
                {
                    lock (_boxLock) { _lastPalmBox = null; }
                    return;
                }

                float dx = boxes[0, bestIndex, 0];
                float dy = boxes[0, bestIndex, 1];
                float dw = boxes[0, bestIndex, 2];
                float dh = boxes[0, bestIndex, 3];

                var anchor = _anchors[bestIndex];

                float cx = dx / 192f + anchor.x;
                float cy = dy / 192f + anchor.y;
                float w = dw / 192f;
                float h = dh / 192f;

                int x1 = (int)((cx - w / 2f) * scale - padLeft);
                int y1 = (int)((cy - h / 2f) * scale - padTop);
                int boxW = (int)(w * scale);
                int boxH = (int)(h * scale);

                lock (_boxLock)
                {
                    _lastPalmBox = new OpenCvSharp.Rect(x1, y1, boxW, boxH);
                }
            }
        }

        private List<(float x, float y)> _anchors;

        private List<(float x, float y)> GenerateAnchors()
        {
            var anchors = new List<(float x, float y)>();
            int[] strides = { 8, 16, 16, 16 };
            int inputSize = 192;
            int layerId = 0;

            while (layerId < strides.Length)
            {
                int lastSameStride = layerId;
                int anchorsPerCell = 0;

                while (lastSameStride < strides.Length && strides[lastSameStride] == strides[layerId])
                {
                    anchorsPerCell += 2;
                    lastSameStride++;
                }

                int stride = strides[layerId];
                int featureMapSize = (int)Math.Ceiling((double)inputSize / stride);

                for (int y = 0; y < featureMapSize; y++)
                {
                    for (int x = 0; x < featureMapSize; x++)
                    {
                        for (int a = 0; a < anchorsPerCell; a++)
                        {
                            float xCenter = (x + 0.5f) / featureMapSize;
                            float yCenter = (y + 0.5f) / featureMapSize;
                            anchors.Add((xCenter, yCenter));
                        }
                    }
                }

                layerId = lastSameStride;
            }

            return anchors;
        }

        private void DetectLandmarks(Mat frame, OpenCvSharp.Rect palmBox)
        {
            int centerX = palmBox.X + palmBox.Width / 2;
            int centerY = palmBox.Y + palmBox.Height / 2 - (int)(palmBox.Height * 0.1f);
            int side = (int)(Math.Max(palmBox.Width, palmBox.Height) * 2.6f);

            int cropX = centerX - side / 2;
            int cropY = centerY - side / 2;

            int x1 = Math.Max(0, cropX);
            int y1 = Math.Max(0, cropY);
            int x2 = Math.Min(frame.Width, cropX + side);
            int y2 = Math.Min(frame.Height, cropY + side);

            if (x2 - x1 <= 0 || y2 - y1 <= 0) return;

            using var cropped = new Mat(frame, new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1));
            using var rgb = new Mat();
            Cv2.CvtColor(cropped, rgb, ColorConversionCodes.BGR2RGB);
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new OpenCvSharp.Size(224, 224));

            var tensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    var pixel = resized.At<Vec3b>(y, x);
                    tensor[0, y, x, 0] = pixel.Item0 / 255f;
                    tensor[0, y, x, 1] = pixel.Item1 / 255f;
                    tensor[0, y, x, 2] = pixel.Item2 / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input_1", tensor)
    };

            using var results = _landmarkDetector.Run(inputs);

            float conf = results.First(r => r.Name == "Identity_1").AsTensor<float>().GetValue(0);
            if (conf < 0.5f)
            {
                lock (_landmarksLock) { _lastLandmarks = null; }
                return;
            }

            var raw = results.First(r => r.Name == "Identity").AsTensor<float>();

            float scaleX = (float)(x2 - x1) / 224f;
            float scaleY = (float)(y2 - y1) / 224f;

            var points = new OpenCvSharp.Point2f[21];
            for (int i = 0; i < 21; i++)
            {
                float lx = raw[0, i * 3];
                float ly = raw[0, i * 3 + 1];
                points[i] = new OpenCvSharp.Point2f(lx * scaleX + x1, ly * scaleY + y1);
            }

            lock (_landmarksLock)
            {
                _lastLandmarks = points;
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _running = false;
            _cameraTask?.Wait();
            _detectionTask?.Wait();
            _capture?.Release();
            _frame?.Dispose();
        }
    }
}
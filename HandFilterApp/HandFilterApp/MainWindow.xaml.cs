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
using System.Runtime.InteropServices;

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
        private int _currentFilterIndex = 0;
        private bool _wasTouching = false;
        private DateTime _lastGestureTime = DateTime.MinValue;
        private readonly object _landmarksLock = new object();
        private List<OpenCvSharp.Point2f[]> _lastLandmarksList = new List<OpenCvSharp.Point2f[]>();
        private readonly object _boxLock = new object();
        private List<OpenCvSharp.Rect> _lastPalmBoxes = new List<OpenCvSharp.Rect>();

        public MainWindow()
        {
            InitializeComponent();
            LoadModels();
            _anchors = GenerateAnchors();
            StartCamera();
        }

        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);

        private void EnableDarkTitleBar()
        {
            var hwnd = new System.Windows.Interop.WindowInteropHelper(this).Handle;
            int useDarkMode = 1;
            DwmSetWindowAttribute(hwnd, 20, ref useDarkMode, sizeof(int));
        }

        protected override void OnSourceInitialized(EventArgs e)
        {
            base.OnSourceInitialized(e);
            EnableDarkTitleBar();
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

            List<OpenCvSharp.Rect> boxes;
            lock (_boxLock) { boxes = _lastPalmBoxes; }

            foreach (var box in boxes)
            {
                Cv2.Rectangle(frameCopy, box, Scalar.LimeGreen, 3);
            }

            List<OpenCvSharp.Point2f[]> landmarksList;
            lock (_landmarksLock) { landmarksList = _lastLandmarksList; }

            foreach (var landmarks in landmarksList)
            {
                foreach (var pt in landmarks)
                {
                    Cv2.Circle(frameCopy, (OpenCvSharp.Point)pt, 5, Scalar.Red, -1);
                }
            }

            ApplyFrameFilter(frameCopy, landmarksList);

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
            rgb.GetArray(out Vec3b[] pixels);
            int idx = 0;
            for (int y = 0; y < 192; y++)
            {
                for (int x = 0; x < 192; x++)
                {
                    var p = pixels[idx++];
                    tensor[0, y, x, 0] = p.Item0 / 255f;
                    tensor[0, y, x, 1] = p.Item1 / 255f;
                    tensor[0, y, x, 2] = p.Item2 / 255f;
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

                    List<OpenCvSharp.Rect> boxes;
                    lock (_boxLock) { boxes = new List<OpenCvSharp.Rect>(_lastPalmBoxes); }

                    var landmarksList = new List<OpenCvSharp.Point2f[]>();
                    foreach (var box in boxes)
                    {
                        var pts = DetectLandmarks(frameCopy, box);
                        if (pts != null) landmarksList.Add(pts);
                    }

                    lock (_landmarksLock) { _lastLandmarksList = landmarksList; }

                    CheckGesture(landmarksList);

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

        private float IoU(OpenCvSharp.Rect a, OpenCvSharp.Rect b)
        {
            var inter = a & b;
            float interArea = inter.Width * inter.Height;
            float unionArea = (a.Width * a.Height) + (b.Width * b.Height) - interArea;
            return unionArea <= 0 ? 0 : interArea / unionArea;
        }

        private void DetectPalm(Mat frame)
        {
            var inputTensor = PreprocessFrame(frame, out float padLeft, out float padTop, out float scale);
            var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input_1", inputTensor)
    };

            var candidates = new List<(float score, OpenCvSharp.Rect box)>();

            using (var results = _palmDetector.Run(inputs))
            {
                var boxes = results.First(r => r.Name == "Identity").AsTensor<float>();
                var scores = results.First(r => r.Name == "Identity_1").AsTensor<float>();

                for (int i = 0; i < scores.Length; i++)
                {
                    float score = Sigmoid(scores.GetValue(i));
                    if (score < 0.5f) continue;

                    float dx = boxes[0, i, 0];
                    float dy = boxes[0, i, 1];
                    float dw = boxes[0, i, 2];
                    float dh = boxes[0, i, 3];

                    var anchor = _anchors[i];
                    float cx = dx / 192f + anchor.x;
                    float cy = dy / 192f + anchor.y;
                    float w = dw / 192f;
                    float h = dh / 192f;

                    int x1 = (int)((cx - w / 2f) * scale - padLeft);
                    int y1 = (int)((cy - h / 2f) * scale - padTop);
                    int boxW = (int)(w * scale);
                    int boxH = (int)(h * scale);

                    candidates.Add((score, new OpenCvSharp.Rect(x1, y1, boxW, boxH)));
                }
            }

            candidates.Sort((a, b) => b.score.CompareTo(a.score));

            var finalBoxes = new List<OpenCvSharp.Rect>();
            foreach (var c in candidates)
            {
                if (finalBoxes.Count >= 2) break;

                bool overlaps = false;
                foreach (var fb in finalBoxes)
                {
                    if (IoU(c.box, fb) > 0.3f) { overlaps = true; break; }
                }

                if (!overlaps) finalBoxes.Add(c.box);
            }

            lock (_boxLock)
            {
                _lastPalmBoxes = finalBoxes;
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

        private OpenCvSharp.Point2f[] DetectLandmarks(Mat frame, OpenCvSharp.Rect palmBox)
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

            if (x2 - x1 <= 0 || y2 - y1 <= 0) return null;

            int padLeft = x1 - cropX;
            int padTop = y1 - cropY;
            int padRight = (cropX + side) - x2;
            int padBottom = (cropY + side) - y2;

            using var rawCrop = new Mat(frame, new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1));
            using var cropped = new Mat();
            Cv2.CopyMakeBorder(rawCrop, cropped, padTop, padBottom, padLeft, padRight, BorderTypes.Constant, Scalar.Black);

            using var rgb = new Mat();
            Cv2.CvtColor(cropped, rgb, ColorConversionCodes.BGR2RGB);
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new OpenCvSharp.Size(224, 224));

            var tensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
            resized.GetArray(out Vec3b[] pixels);
            int idx = 0;
            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    var p = pixels[idx++];
                    tensor[0, y, x, 0] = p.Item0 / 255f;
                    tensor[0, y, x, 1] = p.Item1 / 255f;
                    tensor[0, y, x, 2] = p.Item2 / 255f;
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
                return null;
            }

            var raw = results.First(r => r.Name == "Identity").AsTensor<float>();

            float scaleX = (float)side / 224f;
            float scaleY = (float)side / 224f;

            var points = new OpenCvSharp.Point2f[21];
            for (int i = 0; i < 21; i++)
            {
                float lx = raw[0, i * 3];
                float ly = raw[0, i * 3 + 1];
                points[i] = new OpenCvSharp.Point2f(lx * scaleX + cropX, ly * scaleY + cropY);
            }

            return points;
        }

        private void CheckGesture(List<OpenCvSharp.Point2f[]> landmarksList)
        {
            bool isTouching = false;

            foreach (var landmarks in landmarksList)
            {
                var thumbTip = landmarks[4];
                var pinkyTip = landmarks[20];
                var wrist = landmarks[0];
                var middleBase = landmarks[9];

                float handSize = Distance(wrist, middleBase);
                float touchDistance = Distance(thumbTip, pinkyTip);

                if (touchDistance < handSize * 0.4f)
                {
                    isTouching = true;
                    break;
                }
            }

            if (isTouching && !_wasTouching)
            {
                if ((DateTime.Now - _lastGestureTime).TotalMilliseconds > 800)
                {
                    _currentFilterIndex++;
                    _lastGestureTime = DateTime.Now;
                    System.Diagnostics.Debug.WriteLine($"¡Gesto detectado! Filtro actual: {_currentFilterIndex}");
                }
            }

            _wasTouching = isTouching;
        }

        private float Distance(OpenCvSharp.Point2f a, OpenCvSharp.Point2f b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        private void ApplyFrameFilter(Mat frame, List<OpenCvSharp.Point2f[]> landmarksList)
        {
            if (landmarksList.Count < 2) return;

            var hand1 = landmarksList[0];
            var hand2 = landmarksList[1];

            var (left, right) = hand1[0].X < hand2[0].X ? (hand1, hand2) : (hand2, hand1);

            var indexLeft = left[8];
            var indexRight = right[8];
            var thumbRight = right[4];
            var thumbLeft = left[4];

            var quad = new[]
            {
        (OpenCvSharp.Point)indexLeft,
        (OpenCvSharp.Point)indexRight,
        (OpenCvSharp.Point)thumbRight,
        (OpenCvSharp.Point)thumbLeft
    };

            var boundingRect = Cv2.BoundingRect(quad) & new OpenCvSharp.Rect(0, 0, frame.Width, frame.Height);
            if (boundingRect.Width <= 0 || boundingRect.Height <= 0) return;

            using var roi = new Mat(frame, boundingRect);
            using var mask = Mat.Zeros(roi.Size(), MatType.CV_8UC1).ToMat();

            var shiftedQuad = quad.Select(p => new OpenCvSharp.Point(p.X - boundingRect.X, p.Y - boundingRect.Y)).ToArray();
            Cv2.FillConvexPoly(mask, shiftedQuad, Scalar.White);

            using var filtered = ApplyFilterEffect(roi, _currentFilterIndex);
            filtered.CopyTo(roi, mask);
        }

        //Filtros

        private Mat HeatmapFilter(Mat src)
        {
            using var gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
            var result = new Mat();
            Cv2.ApplyColorMap(gray, result, ColormapTypes.Jet);
            return result;
        }

        private Mat SketchFilter(Mat src)
        {
            using var gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
            using var inverted = new Mat();
            Cv2.BitwiseNot(gray, inverted);
            using var blurred = new Mat();
            Cv2.GaussianBlur(inverted, blurred, new OpenCvSharp.Size(21, 21), 0);
            using var invertedBlur = new Mat();
            Cv2.BitwiseNot(blurred, invertedBlur);
            using var sketchGray = new Mat();
            Cv2.Divide(gray, invertedBlur, sketchGray, 256.0);
            var result = new Mat();
            Cv2.CvtColor(sketchGray, result, ColorConversionCodes.GRAY2BGR);
            return result;
        }

        private Mat PrismFilter(Mat src)
        {
            OpenCvSharp.Mat[] channels = Cv2.Split(src);

            int shiftX = Math.Max(10, src.Width / 20);
            int shiftY = Math.Max(3, src.Height / 100);

            using var mB = new Mat(2, 3, MatType.CV_32F);
            mB.Set(0, 0, 1f); mB.Set(0, 1, 0f); mB.Set(0, 2, -(float)shiftX);
            mB.Set(1, 0, 0f); mB.Set(1, 1, 1f); mB.Set(1, 2, (float)shiftY);

            using var mR = new Mat(2, 3, MatType.CV_32F);
            mR.Set(0, 0, 1f); mR.Set(0, 1, 0f); mR.Set(0, 2, (float)shiftX);
            mR.Set(1, 0, 0f); mR.Set(1, 1, 1f); mR.Set(1, 2, -(float)shiftY);

            using var bShifted = new Mat();
            using var rShifted = new Mat();

            Cv2.WarpAffine(channels[0], bShifted, mB, src.Size(), InterpolationFlags.Linear, BorderTypes.Replicate);
            Cv2.WarpAffine(channels[2], rShifted, mR, src.Size(), InterpolationFlags.Linear, BorderTypes.Replicate);

            var result = new Mat();
            Cv2.Merge(new[] { bShifted, channels[1], rShifted }, result);

            channels[0].Dispose();
            channels[1].Dispose();
            channels[2].Dispose();

            return result;
        }
        private Mat EdgeFilter(Mat src)
        {
            using var gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
            using var edges = new Mat();
            Cv2.Canny(gray, edges, 50, 150);
            var result = new Mat(src.Size(), MatType.CV_8UC3, Scalar.Black);
            result.SetTo(new Scalar(255, 255, 0), edges);
            return result;
        }

        private Mat DuotoneFilter(Mat src, Scalar darkColor, Scalar lightColor)
        {
            using var gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

            using var lutB = new Mat(1, 256, MatType.CV_8UC1);
            using var lutG = new Mat(1, 256, MatType.CV_8UC1);
            using var lutR = new Mat(1, 256, MatType.CV_8UC1);

            for (int i = 0; i < 256; i++)
            {
                float t = i / 255f;
                byte b = (byte)(darkColor.Val0 + (lightColor.Val0 - darkColor.Val0) * t);
                byte g = (byte)(darkColor.Val1 + (lightColor.Val1 - darkColor.Val1) * t);
                byte r = (byte)(darkColor.Val2 + (lightColor.Val2 - darkColor.Val2) * t);
                lutB.Set(0, i, b);
                lutG.Set(0, i, g);
                lutR.Set(0, i, r);
            }

            using var channelB = new Mat();
            using var channelG = new Mat();
            using var channelR = new Mat();
            Cv2.LUT(gray, lutB, channelB);
            Cv2.LUT(gray, lutG, channelG);
            Cv2.LUT(gray, lutR, channelR);

            var result = new Mat();
            Cv2.Merge(new[] { channelB, channelG, channelR }, result);
            return result;
        }

        private Mat GrayscaleFilter(Mat src)
        {
            using var gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
            var result = new Mat();
            Cv2.CvtColor(gray, result, ColorConversionCodes.GRAY2BGR);
            return result;
        }

        private Mat ApplyFilterEffect(Mat roi, int filterIndex)
        {
            switch (((filterIndex % 7) + 7) % 7)
            {
                case 0: return HeatmapFilter(roi);
                case 1: return SketchFilter(roi);
                case 2: return PrismFilter(roi);
                case 3: return EdgeFilter(roi);
                case 4: return DuotoneFilter(roi, new Scalar(180, 50, 220), new Scalar(0, 220, 255));
                case 5: return GrayscaleFilter(roi);
                case 6:
                    {
                        var inverted = new Mat();
                        Cv2.BitwiseNot(roi, inverted);
                        return inverted;
                    }
                default: return roi.Clone();
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
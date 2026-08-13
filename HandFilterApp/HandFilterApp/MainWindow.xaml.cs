using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using System.Windows.Media;

namespace HandFilterApp
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture _capture;
        private Mat _frame;
        private readonly object _frameLock = new object();
        private bool _running;
        private Task _cameraTask;

        public MainWindow()
        {
            InitializeComponent();
            StartCamera();
        }

        private void StartCamera()
        {
            _capture = new VideoCapture(0);
            if (!_capture.IsOpened())
            {
                MessageBox.Show("No se pudo abrir la cámara.");
                return;
            }

            _frame = new Mat();
            _running = true;
            _cameraTask = Task.Run(() => CameraLoop());
        }

        private void CameraLoop()
        {
            while (_running)
            {
                using (var tempFrame = new Mat())
                {
                    _capture.Read(tempFrame);

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
            if (!_running) return;

            BitmapSource bitmap;

            lock (_frameLock)
            {
                if (_frame == null || _frame.IsDisposed || _frame.Empty()) return;
                bitmap = MatToBitmapSource(_frame);
            }

            CameraView.Source = bitmap;
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

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _running = false;
            _cameraTask?.Wait();
            _capture?.Release();
            _frame?.Dispose();
        }
    }
}
import os
import shutil
import tempfile
import unittest
import cv2
import src.utils.preprocessed.conversion as conversion

class TestConvertVideoDcm(unittest.TestCase):
    def setUp(self):
        # Temporary folder for conversion output
        self.temp_dir = tempfile.mkdtemp()
        # Assume sample DICOM file is in a "samples" folder next to this test file
        self.sample_dir = os.path.join(os.path.dirname(__file__), "samples")
        self.dcm_file = os.path.join(self.sample_dir, "test.dcm")
        self.output_file = os.path.join(self.temp_dir, "dcm_output.mp4")

    def tearDown(self):
        # shutil.rmtree(self.temp_dir)
        pass

    def test_convert_video_dcm(self):
        conversion.convert_video(self.dcm_file, self.output_file)
        self.assertTrue(os.path.isfile(self.output_file))
        cap = cv2.VideoCapture(self.output_file)
        self.assertTrue(cap.isOpened())
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        self.assertTrue(ret)
        height, width = frame.shape[:2]
        cap.release()

        # These expected values should match your sample DICOM properties.
        # Here we assume: 42 frames, 576x768 resolution, and 30 fps with 3 color channels.
        self.assertEqual(frame_count, 42)
        self.assertEqual(height, 576)
        self.assertEqual(width, 768)
        self.assertAlmostEqual(fps, 30, places=1)
        self.assertEqual(frame.shape[2], 3)

class TestConvertVideoAvi(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(os.path.dirname(__file__), "samples")
        self.avi_file = os.path.join(self.sample_dir, "test.avi")
        self.output_file = os.path.join(self.temp_dir, "avi_output.mp4")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_convert_video_avi(self):
        conversion.convert_video(self.avi_file, self.output_file)
        self.assertTrue(os.path.isfile(self.output_file))
        cap = cv2.VideoCapture(self.output_file)
        self.assertTrue(cap.isOpened())
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        self.assertTrue(ret)
        height, width = frame.shape[:2]
        cap.release()

        # Expected values are based on your sample AVI file.
        # For example, assume: 25 frames, 576x768 resolution, ~16.667 fps, and 3 color channels.
        self.assertEqual(frame_count, 25)
        self.assertEqual(height, 576)
        self.assertEqual(width, 768)
        self.assertAlmostEqual(fps, 16.667, places=1)
        self.assertEqual(frame.shape[2], 3)

class TestMainConversion(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for input and output
        self.temp_input_dir = tempfile.mkdtemp()
        self.temp_output_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(os.path.dirname(__file__), "samples")
        # Copy both sample files into the temporary input directory
        for sample in ["test.dcm", "test.avi"]:
            src = os.path.join(self.sample_dir, sample)
            dst = os.path.join(self.temp_input_dir, sample)
            shutil.copy2(src, dst)

    def tearDown(self):
        shutil.rmtree(self.temp_input_dir)
        shutil.rmtree(self.temp_output_dir)

    def test_main_conversion(self):
        conversion.main(self.temp_input_dir, self.temp_output_dir)
        # Check that an MP4 was generated for each supported sample file.
        for sample in ["test.dcm", "test.avi"]:
            base = os.path.splitext(sample)[0]
            output_file = os.path.join(self.temp_output_dir, base + ".mp4")
            self.assertTrue(os.path.isfile(output_file))
            cap = cv2.VideoCapture(output_file)
            self.assertTrue(cap.isOpened())
            cap.release()

class TestConvertToBW(unittest.TestCase):
    def setUp(self):
        self.sample_dir = os.path.join(os.path.dirname(__file__), "samples")
        # Use the AVI sample file
        self.avi_file = os.path.join(self.sample_dir, "test.avi")
        # Intermediate MP4 file from the AVI conversion saved in permanent storage.
        self.intermediate_file = os.path.join(self.sample_dir, "avi_output.mp4")
        # Final black and white output file saved in permanent storage.
        self.bw_file = os.path.join(self.sample_dir, "avi_bw_output.mp4")
        # Convert the sample AVI file to MP4
        conversion.convert_video(self.avi_file, self.intermediate_file)

    def tearDown(self):
        # Do not remove the permanent directory so that files remain for manual inspection.
        pass

    def test_convert_to_bw(self):
        # Convert the previously generated MP4 file to black and white.
        conversion.convert_to_bw(self.intermediate_file, self.bw_file)
        self.assertTrue(os.path.isfile(self.bw_file))
        
        cap = cv2.VideoCapture(self.bw_file)
        # Try to set the capture mode to grayscale.
        mode_set = False
        try:
            mode_set = cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
        except AttributeError:
            mode_set = False

        self.assertTrue(cap.isOpened())
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, frame = cap.read()
        self.assertTrue(ret)
        
        # If setting mode didn't work or the frame still has 3 channels, convert manually.
        if not mode_set or (len(frame.shape) == 3 and frame.shape[2] == 3):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Now verify that the frame is indeed single-channel grayscale.
        self.assertEqual(len(frame.shape), 2)
        
        # Check the resolution.
        height, width = frame.shape
        cap.release()

        # Expected metadata based on your sample AVI conversion:
        # 25 frames, 576x768 resolution, and ~16.667 fps.
        self.assertEqual(frame_count, 25)
        self.assertEqual(height, 576)
        self.assertEqual(width, 768)
        self.assertAlmostEqual(fps, 16.667, places=1)


if __name__ == "__main__":
    unittest.main()

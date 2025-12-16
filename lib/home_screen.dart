import 'dart:io';
import 'dart:math';
import 'package:facerecognition_app/exceptions.dart';  // Add this import for the custom exceptions
import 'package:facerecognition_app/tflite_service.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {

  double cosineSimilarity(List<double> e1, List<double> e2) {
    double dot = 0, norm1 = 0, norm2 = 0;
    for (int i = 0; i < e1.length; i++) {
      dot += e1[i] * e2[i];
      norm1 += e1[i] * e1[i];
      norm2 += e2[i] * e2[i];
    }
    return dot / (sqrt(norm1) * sqrt(norm2));  // Closer to 1 = better match
  }

  File? _profileImage;
  File? _verificationSelfie;

  List<double>? _profileEmbeddings;
  List<double>? _verificationEmbeddings;

  final ImagePicker _picker = ImagePicker();
  final TFLiteService _tfliteService = TFLiteService();

  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _tfliteService.loadModel();
  }

  // ---------------- PROFILE IMAGE ----------------
  Future<void> _pickProfileImage() async {
    _showInfoDialog(
      title: "Select Profile Photo",
      message:
      "Choose a clear photo of your face from the gallery.\n\nThis will be used as your reference image.",
    );

    final pickedFile = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 90,
    );

    if (pickedFile != null) {
      setState(() {
        _profileImage = File(pickedFile.path);
      });

      try {
        _showLoading("Processing...");
        _profileEmbeddings = await _tfliteService.getEmbeddings(_profileImage!);
        if (_profileEmbeddings == null) throw Exception("No face detected");
        _hideLoading();
        _showInfoDialog(title: "Profile Set", message: "Ready for verification!");
      } on MultipleFacesException catch (e) {
        _hideLoading();
        _showInfoDialog(title: "Multiple Faces", message: e.message);
        setState(() => _profileImage = null);  // Reset on failure
      } on NoFaceException catch (e) {
        _hideLoading();
        _showInfoDialog(title: "No Face Detected", message: e.message);
        setState(() => _profileImage = null);  // Reset on failure
      } catch (e) {
        _hideLoading();
        _showInfoDialog(title: "Error", message: "Failed to process: $e. Try a clearer photo.");
        setState(() => _profileImage = null);  // Reset on failure
      }
    }
  }

  // ---------------- VERIFICATION SELFIE ----------------
  Future<void> _takeVerificationSelfie() async {
    if (_profileImage == null) {
      _showInfoDialog(
        title: "Profile Required",
        message:
        "Please select your profile image before verification.",
      );
      return;
    }

    _showInfoDialog(
      title: "Face Verification",
      message:
      "We will now take a selfie using the front camera.\n\nPlease look straight into the camera.",
    );

    final pickedFile = await _picker.pickImage(
      source: ImageSource.camera,
      preferredCameraDevice: CameraDevice.front,
      imageQuality: 90,
    );

    if (pickedFile != null) {
      setState(() {
        _verificationSelfie = File(pickedFile.path);
      });

      try {
        _showLoading("Verifying your face...");
        _verificationEmbeddings = await _tfliteService.getEmbeddings(_verificationSelfie!);
        if (_verificationEmbeddings == null) throw Exception("No face detected");
        _hideLoading();
        _compareFaces();
      } on MultipleFacesException catch (e) {
        _hideLoading();
        _showResultDialog(title: "Multiple Faces", message: e.message, success: false);
      } on NoFaceException catch (e) {
        _hideLoading();
        _showResultDialog(title: "No Face Detected", message: e.message, success: false);
      } catch (e) {
        _hideLoading();
        _showResultDialog(title: "Processing Error", message: "Failed to process selfie: $e. Try again.", success: false);
      }
    }
  }

  // ---------------- FACE COMPARISON ----------------
  void _compareFaces() {
    if (_profileEmbeddings == null || _verificationEmbeddings == null) {
      _showResultDialog(title: "Error", message: "Failed to extract faces. Ensure clear photos.", success: false);
      return;
    }

    try {
      final similarity = cosineSimilarity(_profileEmbeddings!, _verificationEmbeddings!);
      final threshold = 0.5;  // Lowered from 0.6 for more leniency on minor changes like accessories—tune based on tests

      String message;
      if (similarity > threshold) {
        message = "Match confidence: ${(similarity * 100).toStringAsFixed(1)}%—Verified!";
      } else {
        message = "Similarity: ${(similarity * 100).toStringAsFixed(1)}%. Tips:\n"
            "• Match your profile pic's look (e.g., glasses off, clean-shaven for best results).\n"
            "• Straight-on angle, neutral face, even light.\n"
            "• If close (like 50%+), retry without accessories.";
      }

      _showResultDialog(
        title: similarity > threshold ? "Verified!" : "Needs Better Match",
        message: message,
        success: similarity > threshold,
      );
    } catch (e) {
      _showResultDialog(title: "Error", message: "Comparison failed: $e", success: false);
    }
  }

  // ---------------- UI HELPERS ----------------
  void _showLoading(String message) {
    setState(() => _isProcessing = true);
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        content: Row(
          children: [
            const CircularProgressIndicator(),
            const SizedBox(width: 20),
            Expanded(child: Text(message)),
          ],
        ),
      ),
    );
  }

  void _hideLoading() {
    if (_isProcessing) {
      Navigator.of(context).pop();
      _isProcessing = false;
    }
  }

  void _showInfoDialog({required String title, required String message}) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            child: const Text("OK"),
            onPressed: () => Navigator.pop(context),
          )
        ],
      ),
    );
  }

  void _showResultDialog({
    required String title,
    required String message,
    required bool success,
  }) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Row(
          children: [
            Icon(
              success ? Icons.check_circle : Icons.error,
              color: success ? Colors.green : Colors.red,
            ),
            const SizedBox(width: 10),
            Text(title),
          ],
        ),
        content: Text(message),
        actions: [
          TextButton(
            child: const Text("OK"),
            onPressed: () => Navigator.pop(context),
          )
        ],
      ),
    );
  }

  // ---------------- UI ----------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Face Recognition"),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            _imageCard(
              title: "Profile Image",
              image: _profileImage,
              icon: Icons.person,
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              icon: const Icon(Icons.photo_library),
              label: const Text("Select Profile Image"),
              onPressed: _pickProfileImage,
            ),
            const SizedBox(height: 40),
            _imageCard(
              title: "Verification Selfie",
              image: _verificationSelfie,
              icon: Icons.camera_alt,
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              icon: const Icon(Icons.verified_user),
              label: const Text("Verify Face"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
              ),
              onPressed: _takeVerificationSelfie,
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              icon: const Icon(Icons.refresh),
              label: const Text("Reset"),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.grey),
              onPressed: () {
                setState(() {
                  _profileImage = null;
                  _verificationSelfie = null;
                  _profileEmbeddings = null;
                  _verificationEmbeddings = null;
                });
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _imageCard({
    required String title,
    required File? image,
    required IconData icon,
  }) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Container(
        padding: const EdgeInsets.all(12),
        width: double.infinity,
        child: Column(
          children: [
            Text(title,
                style:
                const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            image == null
                ? Icon(icon, size: 80, color: Colors.grey)
                : ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.file(image, height: 160),
            ),
          ],
        ),
      ),
    );
  }
}
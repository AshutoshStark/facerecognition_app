import 'dart:io';
import 'package:facerecognition_app/exceptions.dart';  // Add this import and remove the duplicate class definitions below
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'dart:developer' as developer;

// Remove these duplicate definitions:
// class MultipleFacesException implements Exception {
//   final String message;
//   MultipleFacesException(this.message);
// }
//
// class NoFaceException implements Exception {
//   final String message;
//   NoFaceException(this.message);
// }

Future<img.Image?> detectAndCropFace(File imageFile) async {
  print('Detecting face in: ${imageFile.path}');

  final inputImage = InputImage.fromFile(imageFile);
  final detector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.fast,
      enableLandmarks: false,
      enableClassification: false,
    ),
  );

  try {
    final faces = await detector.processImage(inputImage);
    developer.log('Detected ${faces.length} faces');

    if (faces.isEmpty) {
      // Log image details for debugging
      final bytes = await imageFile.readAsBytes();
      final original = img.decodeImage(bytes);
      if (original != null) {
        developer.log('Image size: ${original.width}x${original.height}');
        developer.log('Potential issue: Face may be too small (<100px) or angled/occluded. Try a closer crop.');
      } else {
        developer.log('Failed to decode image—check format (JPG/PNG preferred)');
      }
      throw NoFaceException('No face detected in the image. Please use a photo with exactly one clear face.');
    }

    if (faces.length > 1) {
      developer.log('Multiple faces detected: ${faces.length}. Rejecting image.');
      throw MultipleFacesException('Multiple faces detected (${faces.length}). Please use a photo with exactly one face.');
    }

    final face = faces.first.boundingBox;
    developer.log('Face bounds: left=${face.left}, top=${face.top}, width=${face.width}, height=${face.height}');

    final original = img.decodeImage(await imageFile.readAsBytes())!;

    const double margin = 0.3;

    final x = (face.left - face.width * margin).clamp(0.0, original.width.toDouble()).toInt();
    final y = (face.top - face.height * margin).clamp(0.0, original.height.toDouble()).toInt();
    final w = (face.width * (1 + 2 * margin)).clamp(0.0, (original.width - x).toDouble()).toInt();
    final h = (face.height * (1 + 2 * margin)).clamp(0.0, (original.height - y).toDouble()).toInt();

    final cropped = img.copyCrop(original, x: x, y: y, width: w, height: h);
    final resized = img.copyResize(cropped, width: 112, height: 112);
    developer.log('Cropped & resized successfully to 112x112');

    return resized;
  } catch (e) {
    developer.log('Detection error: $e');
    rethrow;  // Propagate the specific exception
  } finally {
    detector.close();
  }
}
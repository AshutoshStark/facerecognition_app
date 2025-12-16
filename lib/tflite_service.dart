import 'dart:io';
import 'dart:typed_data';
import 'package:facerecognition_app/exceptions.dart';  // For custom exceptions
import 'package:facerecognition_app/face_utils.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteService {
  Uint8List? _modelBytes;  // Store bytes for isolate serialization

  Future<void> loadModel() async {
    try {
      _modelBytes = (await rootBundle.load('assets/mobilefacenet.tflite')).buffer.asUint8List();
      debugPrint('Model loaded successfully (size: ${_modelBytes!.length} bytes)');
    } catch (e) {
      debugPrint('Failed to load model: $e');
      rethrow;  // Bubble up for UI handling
    }
  }

  Future<List<double>?> getEmbeddings(File imageFile) async {
    if (_modelBytes == null) {
      throw Exception('Model not loaded. Call loadModel() first.');
    }

    try {
      // Do face detection and cropping on main thread
      final faceImage = await detectAndCropFace(imageFile);
      if (faceImage == null) return null;

      // Pass cropped image bytes to isolate for TFLite
      final imageBytes = Uint8List.fromList(img.encodePng(faceImage));
      return await compute(_getEmbeddingsFromImage, {'imageBytes': imageBytes, 'modelBytes': _modelBytes!});
    } on MultipleFacesException {
      rethrow;  // Propagate to UI (no 'e' variable needed)
    } on NoFaceException {
      rethrow;  // Propagate to UI (no 'e' variable needed)
    } catch (e) {
      debugPrint('General detection error: $e');
      rethrow;  // Propagate general errors
    }
  }

  static List<double>? _getEmbeddingsFromImage(Map<String, Uint8List> params) {
    try {
      final imageBytes = params['imageBytes']!;
      final modelBytes = params['modelBytes']!;
      final faceImage = img.decodeImage(imageBytes)!;

      // Input tensor: [1, 112, 112, 3], normalized to [-1, 1]
      final input = Float32List(1 * 112 * 112 * 3);
      int index = 0;
      // FIXED: Use Pixel properties directly (r, g, b are num in [0, 255])
      for (int y = 0; y < faceImage.height; y++) {
        for (int x = 0; x < faceImage.width; x++) {
          final pixel = faceImage.getPixel(x, y);
          // Normalize RGB to [-1, 1]
          input[index++] = ((pixel.r / 255.0) * 2) - 1;
          input[index++] = ((pixel.g / 255.0) * 2) - 1;
          input[index++] = ((pixel.b / 255.0) * 2) - 1;
        }
      }
      final inputTensor = input.reshape([1, 112, 112, 3]);

      // Load interpreter in isolate
      final interpreter = Interpreter.fromBuffer(modelBytes);

      // Output: [1, 192] to match MobileFaceNet
      final output = List.filled(1 * 192, 0.0).reshape([1, 192]);
      interpreter.run(inputTensor, output);
      interpreter.close();

      return List<double>.from(output[0]);
    } catch (e) {
      debugPrint('Embedding error: $e');
      return null;
    }
  }
}
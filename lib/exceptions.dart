// Custom exceptions for face detection
class NoFaceException implements Exception {
  final String message;
  NoFaceException([this.message = 'No face detected in the image.']);

  @override
  String toString() => 'NoFaceException: $message';
}

class MultipleFacesException implements Exception {
  final String message;
  MultipleFacesException([this.message = 'Multiple faces detected. Please ensure only one face is visible.']);

  @override
  String toString() => 'MultipleFacesException: $message';
}
import React, { useState, useRef, useCallback } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FiUpload, FiCamera, FiImage, FiX } from "react-icons/fi";
import { BiWebcam } from "react-icons/bi";
import Webcam from "react-webcam";

const Home = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [useWebcam, setUseWebcam] = useState(false);
  const webcamRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
      setUseWebcam(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
      setUseWebcam(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const toggleWebcam = () => {
    setUseWebcam(!useWebcam);
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const captureImage = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setPreview(imageSrc);

      // Convert base64 to blob
      fetch(imageSrc)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "webcam-capture.png", {
            type: "image/png",
          });
          setSelectedFile(file);
        });
    }
  }, [webcamRef]);

  const handleSubmit = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("image", selectedFile);

    setIsProcessing(true);
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/detect-emotion", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResult(response.data);
    } catch (err) {
      console.error("Error processing image:", err);
      setError("Failed to process image. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-white dark:bg-slate-900">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-800 dark:text-white mb-4">
            Emotion Detection
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Upload an image or use your webcam to detect emotions in faces.
          </p>
        </div>

        <div className="max-w-4xl mx-auto bg-white dark:bg-slate-800 rounded-xl p-8 shadow-xl">
          {/* Source selection buttons */}
          <div className="flex justify-center mb-8 space-x-4">
            <button
              onClick={() => {
                setUseWebcam(false);
                setSelectedFile(null);
                setPreview(null);
                setResult(null);
              }}
              className={`px-5 py-2.5 rounded-lg font-medium flex items-center 
              ${
                !useWebcam
                  ? "bg-blue-600 text-white"
                  : "bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200"
              }`}
            >
              <FiUpload className="mr-2" />
              Upload Image
            </button>
            <button
              onClick={toggleWebcam}
              className={`px-5 py-2.5 rounded-lg font-medium flex items-center
              ${
                useWebcam
                  ? "bg-blue-600 text-white"
                  : "bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200"
              }`}
            >
              <BiWebcam className="mr-2" />
              Use Webcam
            </button>
          </div>

          {/* Main content area */}
          <div className="bg-slate-100 dark:bg-slate-700 rounded-xl p-6">
            {useWebcam ? (
              <div className="space-y-4">
                <div className="relative rounded-lg overflow-hidden bg-black aspect-video max-h-96 mx-auto">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/png"
                    className="w-full h-full object-contain mx-auto"
                  />
                </div>
                {!preview ? (
                  <div className="flex justify-center">
                    <motion.button
                      whileTap={{ scale: 0.95 }}
                      onClick={captureImage}
                      className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition duration-300 flex items-center"
                    >
                      <FiCamera className="mr-2" />
                      Capture Image
                    </motion.button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="relative rounded-lg overflow-hidden max-h-96 mx-auto">
                      <img
                        src={preview}
                        alt="Captured"
                        className="max-h-96 mx-auto rounded-lg object-contain"
                      />
                    </div>
                    <div className="flex justify-center space-x-3">
                      <button
                        onClick={() => setPreview(null)}
                        className="bg-slate-300 dark:bg-slate-600 hover:bg-slate-400 dark:hover:bg-slate-500 text-slate-700 dark:text-white px-6 py-3 rounded-lg font-medium transition"
                      >
                        Recapture
                      </button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleSubmit}
                        disabled={isProcessing}
                        className={`px-6 py-3 rounded-lg font-medium text-white transition
                          ${
                            isProcessing
                              ? "bg-slate-500 cursor-not-allowed"
                              : "bg-blue-600 hover:bg-blue-700"
                          }`}
                      >
                        {isProcessing ? (
                          <span className="flex items-center justify-center">
                            <svg
                              className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                              xmlns="http://www.w3.org/2000/svg"
                              fill="none"
                              viewBox="0 0 24 24"
                            >
                              <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                              ></circle>
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                              ></path>
                            </svg>
                            Processing...
                          </span>
                        ) : (
                          <span className="flex items-center justify-center">
                            <FiImage className="mr-2" />
                            Detect Emotions
                          </span>
                        )}
                      </motion.button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div
                className={`border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-8 text-center
                  ${preview ? "bg-slate-50 dark:bg-slate-800" : ""}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                {!preview ? (
                  <div className="flex flex-col items-center">
                    <div className="w-20 h-20 mb-4 bg-slate-200 dark:bg-slate-700 rounded-full flex items-center justify-center">
                      <FiUpload className="text-3xl text-slate-500 dark:text-slate-400" />
                    </div>
                    <p className="text-lg mb-6 text-slate-600 dark:text-slate-300">
                      Drag and drop your image here or click to browse
                    </p>
                    <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition duration-300 flex items-center">
                      <FiCamera className="mr-2" />
                      Select Image
                      <input
                        type="file"
                        className="hidden"
                        accept="image/*"
                        onChange={handleFileChange}
                      />
                    </label>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="relative">
                      <img
                        src={preview}
                        alt="Preview"
                        className="max-h-96 mx-auto rounded-lg object-contain"
                      />
                      <button
                        onClick={() => {
                          setPreview(null);
                          setSelectedFile(null);
                          setResult(null);
                        }}
                        className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 w-8 h-8 rounded-full flex items-center justify-center text-white"
                      >
                        <FiX />
                      </button>
                    </div>

                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleSubmit}
                      disabled={isProcessing}
                      className={`w-full py-3 rounded-lg font-medium text-white
                        ${
                          isProcessing
                            ? "bg-slate-500 cursor-not-allowed"
                            : "bg-blue-600 hover:bg-blue-700"
                        }`}
                    >
                      {isProcessing ? (
                        <span className="flex items-center justify-center">
                          <svg
                            className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            ></circle>
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                          </svg>
                          Processing...
                        </span>
                      ) : (
                        <span className="flex items-center justify-center">
                          <FiImage className="mr-2" />
                          Detect Emotions
                        </span>
                      )}
                    </motion.button>
                  </div>
                )}
              </div>
            )}
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6 p-4 bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-800 rounded-lg text-center text-red-700 dark:text-red-300"
            >
              {error}
            </motion.div>
          )}

          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="mt-8 bg-slate-100 dark:bg-slate-700 p-6 rounded-xl"
            >
              <h2 className="text-2xl font-bold mb-6 text-center text-slate-800 dark:text-white">
                Detection Results
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-md">
                  <h3 className="text-lg font-medium mb-3 text-slate-700 dark:text-slate-200">
                    Processed Image
                  </h3>
                  <div className="bg-slate-100 dark:bg-slate-900 p-2 rounded-lg">
                    <img
                      src={result?.image_with_emotions || preview}
                      alt="Result"
                      className="rounded-lg w-full object-contain"
                    />
                  </div>
                </div>
                <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-md">
                  <h3 className="text-lg font-medium mb-3 text-slate-700 dark:text-slate-200">
                    Detected Emotions
                  </h3>
                  {result?.emotions?.length > 0 ? (
                    <ul className="space-y-3">
                      {result.emotions.map((item, index) => (
                        <li
                          key={index}
                          className="bg-slate-50 dark:bg-slate-700 p-4 rounded-lg flex items-center"
                        >
                          <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center mr-3 text-white font-medium">
                            {index + 1}
                          </div>
                          <div className="flex-1">
                            <span className="block font-medium text-slate-800 dark:text-white">
                              {item.emotion}
                            </span>
                            <div className="w-full bg-slate-200 dark:bg-slate-600 h-2 rounded-full mt-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full"
                                style={{
                                  width: `${(item.confidence * 100).toFixed(
                                    1
                                  )}%`,
                                }}
                              />
                            </div>
                            <span className="text-sm text-slate-500 dark:text-slate-400 mt-1 block">
                              Confidence: {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="bg-slate-50 dark:bg-slate-700 p-6 rounded-lg text-center text-slate-500 dark:text-slate-400">
                      No emotions detected in the image.
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </div>

        <div className="mt-12 text-center">
          <p className="text-slate-500 dark:text-slate-400 text-sm">
            Developed by Atharva | Emotion Detection Project
          </p>
        </div>
      </div>
    </div>
  );
};

export default Home;

// =============================
//  CAMERA INITIALISATION
// =============================
const video = document.getElementById("cameraFeed");
const canvas = document.getElementById("snapCanvas");
const captureBtn = document.getElementById("captureBtn");
const cameraForm = document.getElementById("cameraForm");
const cameraImageInput = document.getElementById("cameraImage");

// Ask for camera access
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        alert("Camera access denied or not available.");
        console.error(err);
    });

// =============================
//  CAPTURE IMAGE FROM CAMERA
// =============================
captureBtn.addEventListener("click", () => {

    let context = canvas.getContext("2d");

    // Draw video frame onto canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64 PNG
    let base64Image = canvas.toDataURL("image/png");

    // Save to hidden input
    cameraImageInput.value = base64Image;

    // Submit form to backend
    cameraForm.submit();
});

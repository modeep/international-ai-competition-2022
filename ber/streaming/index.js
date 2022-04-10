const socket = io("http://192.168.0.47:5000");

socket.on("changecloseup", closeup);

// v >= 0 && v <= 1
function closeup(v) {
    document.documentElement.style.transform = `scale(${Math.pow(v + 1, 2)})`;
}
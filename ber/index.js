const socket = io("http://192.168.0.47:5000");

function startmove(v) {
    socket.emit("startmove", v);
}

function endmove() {
    socket.emit("endmove");
}

function startcloseup(v) {
    socket.emit("startcloseup", v);
}

function endcloseup() {
    socket.emit("endcloseoup");
}

const up = document.getElementById("up");
const down = document.getElementById("down");
const left = document.getElementById("left");
const right = document.getElementById("right");

let verticalDirection;
let horizontalDirection;
let vibrateInterval;
let vibrateCount = 0;

function startVibrate(duration) {
    window.navigator.vibrate(duration);
}

function stopVibrate() {
    vibrateCount -= 1;
    
    if(vibrateCount <= 0) {
        clearInterval(vibrateInterval);
        vibrateInterval = null;
    }
}

function startPeristentVibrate(duration, interval) {
    if(!vibrateInterval) {
        vibrateInterval = setInterval(function() {
            startVibrate(duration);
        }, interval);
    }
    vibrateCount += 1;
}

function verticalTouchstart(e) {
    e.currentTarget.classList.add("buttonActive");
    startPeristentVibrate(100, 100);
    if(verticalDirection) {
        document.getElementById(verticalDirection).classList.remove("buttonActive");
        endcloseup();
    }
    const direction = e.currentTarget.getAttribute("id");
    verticalDirection = direction;
    startcloseup(direction === "up" ? 1 : -1);
}

function verticalTouchend(e) {
    e.currentTarget.classList.remove("buttonActive");
    stopVibrate();
    if(verticalDirection === e.currentTarget.getAttribute("id")) {
        verticalDirection = null;
        endcloseup();
    }
}

function horizontalTouchstart(e) {
    e.currentTarget.classList.add("buttonActive");
    startPeristentVibrate(100, 100);
    if(horizontalDirection) {
        document.getElementById(horizontalDirection).classList.remove("buttonActive");
        endcloseup();
    }
    const direction = e.currentTarget.getAttribute("id");
    horizontalDirection = direction;
    startmove(direction === "left" ? 1 : -1);
}

function horizontalTouchend(e) {
    e.currentTarget.classList.remove("buttonActive");
    stopVibrate();
    if(horizontalDirection === e.currentTarget.getAttribute("id")) {
        horizontalDirection = null;
        endmove();
    }
}

up.addEventListener("touchstart", verticalTouchstart);
up.addEventListener("touchend", verticalTouchend);
down.addEventListener("touchstart", verticalTouchstart);
down.addEventListener("touchend", verticalTouchend);
left.addEventListener("touchstart", horizontalTouchstart);
left.addEventListener("touchend", horizontalTouchend);
right.addEventListener("touchstart", horizontalTouchstart);
right.addEventListener("touchend", horizontalTouchend);

window.addEventListener("contextmenu", e => e.preventDefault());

window.onclick = () => {
    document.documentElement.requestFullscreen();
}
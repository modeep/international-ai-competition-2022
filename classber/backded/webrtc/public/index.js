window.onload = () => {
    document.getElementById('my-button').onclick = () => {
        init();
    }
}

async function init() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    // srcObject 카메라 세팅
    document.getElementById("video").srcObject = stream;
    const peer = createPeer();
    // 원격 유저에게 스트리밍을 시작하도록 함.
    stream.getTracks().forEach(track => peer.addTrack(track, stream));
}


function createPeer() {
    const peer = new RTCPeerConnection({
        iceServers: [
            {
                urls: "stun:stun.stunprotocol.org"
            }
        ]
    });
    peer.onnegotiationneeded = () => handleNegotiationNeededEvent(peer);

    return peer;
}

async function handleNegotiationNeededEvent(peer) {
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    const payload = {
        sdp: peer.localDescription
    };

    // axios로 post 방식의 http 통신을 한다.
    // 위에서 준비한 SDP LocalDescription을 인자로 넘겨준다.
    const { data } = await axios.post('/broadcast', payload);
    const desc = new RTCSessionDescription(data.sdp);
    peer.setRemoteDescription(desc).catch(e => console.log(e));
}
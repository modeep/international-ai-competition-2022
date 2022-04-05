import express from "express";
import bodyParser from 'body-parser';
import webrtc from 'wrtc';

const app = express();

let senderStream;

// app.use - 미들웨어 설정
app.use(express.static('public')); // 이미지, CSS 파일 및 JS 파일 같은 정적 파일 제공
app.use(bodyParser.json()); // req.body를 json 형식으로 읽을 수 있음
app.use(bodyParser.urlencoded({ extended: true })); // 객체 형태로 전달된 데이터 내에서 또다른 중첩인 객체를 허용

app.post("/consumer", async ({ body }, res) => {
    const peer = new webrtc.RTCPeerConnection({
        iceServers: [
            {
                urls: "stun:stun.stunprotocol.org"
            }
        ]
    });
    const desc = new webrtc.RTCSessionDescription(body.sdp);
    await peer.setRemoteDescription(desc);
    // addTrack 다른 유저에게 전송될 트랙들의 묶음에 신규 트핵을 추가한다.
    senderStream.getTracks().forEach(track => peer.addTrack(track, senderStream));
    const answer = await peer.createAnswer();
    await peer.setLocalDescription(answer);
    const payload = {
        sdp: peer.localDescription
    }

    res.json(payload);
});

app.post('/broadcast', async ({ body }, res) => {
    // 로컬 컴퓨터와 원격 피어 간의 WebRTC 연결을 담당
    // 연결을 유지하고 연결 상태를 모니터링하며 더 이상 연결이 필요하지 않을 경우 연결을 종료
    const peer = new webrtc.RTCPeerConnection({
        iceServers: [
            {
                urls: "stun:stun.stunprotocol.org"
            }
        ]
    });
    // RTCPeerConnection에 트랙이 등록됨을 알려주는 track이 발생하면 호출되는 함수를 지정하는 event handler
    peer.ontrack = (e) => handleTrackEvent(e, peer);
    // 연결의 한 쪽 끝 또는 잠재적 연결과 어떻게 구성되었는지 기술하는 인터페이스
    // offer와 answer 부분의 협상 프로세스의 어느 부분을 설명하는지 나타내는 설명 유형과 세션의 SDP 설명자로 구성된다.
    const desc = new webrtc.RTCSessionDescription(body.sdp);
    await peer.setRemoteDescription(desc);
    const answer = await peer.createAnswer();
    await peer.setLocalDescription(answer);
    const payload = {
        sdp: peer.localDescription
    }

    res.json(payload);
});

function handleTrackEvent(e, peer) {
    senderStream = e.streams[0];
};


app.listen(5000, () => console.log('server started'));
# smart_cat_door

```mermaid
graph TD
        A(["Start"])
        A --> B{"Check override switch"}
        B --Enabled--> C["Door opened"]
        C --> J["Wait 10 seconds"]
        J --> B
        B --Disabled--> D{"Camera frame available?"}
        D --Noo--> B
        D --Yes--> E{"Prey detected?"}
        E --Yes--> F["Close door"]
        E --Noo--> G["Open door"]
        G --> H["Wait 10 seconds"]
        F --> I["Wait 5 minutes"]
        H --> B
        I --> B
```

## Viewing the camera feed remotely

Camera is on a link-local network wired only to the Pi's `eth0` — not reachable from another machine directly, even over VPN. Tunnel through the Pi over SSH:

```bash
ssh -L 8554:169.254.1.1:554 carbotton@<pi-address> -N
```

Then, from the other machine — force TCP transport, since SSH `-L` only tunnels TCP and RTSP video normally goes over UDP (VLC's client falls back to UDP regardless, breaking the tunnel; `ffplay` respects the flag):

```bash
ffplay -rtsp_transport tcp rtsp://localhost:8554/live/0/MAIN
```


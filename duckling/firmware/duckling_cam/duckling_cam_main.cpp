// duckling_cam_main.cpp - de code van duckling_cam.ino (in een .cpp: geen automatische prototypes nodig).
#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ArduinoOTA.h>
#include <HTTPUpdateServer.h>
#include "vision.h"
#include "duck_protocol.h"

// ------------------------------------------------------------------ instellingen
#define UART_RX 13          // <- RP2350 GPIO16
#define UART_TX 14          // -> RP2350 GPIO17
#define UART_BAUD 921600
#define SEND_HZ 20
#define WIFI_AAN true       // false: nooit wifi. true: wifi zoals de RP2350 zegt (commando "wifi aan|uit"),
                            // zonder RP2350 (los op tafel) altijd aan
#define AP_NAAM "Duckling-cam"
#define AP_WW "duckling1"

// AI-Thinker ESP32-CAM pinnen
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define FLASH_LED 4
#define LED_CH 4            // LEDC-kanaal 4 (timer 2): kanaal 0/timer 0 is de klok van de camera
#define LED_HOLD 6          // beelden negeren na een verandering van de led (de belichting past zich aan)

static Vision vis;
static HeadInfo head;
static uint32_t head_t = 0;           // ms van het laatste RP-pakket
static SemaphoreHandle_t mtx;
static uint8_t snap_gray[VW * VH], snap_diff[VW * VH];
static VisionOut snap_out;
static float fps = 0;
static uint16_t sensor_pid = 0;
static volatile int last_w = 0;   // breedte van het laatste camerabeeld   // 0x26 = OV2640, 0x3660 = OV3660, 0x5640 = OV5640
static Preferences prefs;
static WebServer web(80);
static HTTPUpdateServer web_updater;   // 2026-11-03: firmware uploaden via de browser (geen firewall-gedoe)
static PacketReader<RpPacket, RP_H0, RP_H1> rp_reader;
static uint8_t seq = 0;
static volatile bool rp_wifi = true;     // wat de RP2350 wil
static volatile uint8_t rp_lamp = 0, lamp_now = 255;
static volatile int led_hold = 0;
static bool ota_started = false; static volatile bool ota_busy = false;
static bool wifi_on = false, web_started = false;

// ------------------------------------------------------------------ instellingen bewaren
#define CFG_FIELDS(X) X(diff_thr) X(min_energy) X(i1) X(i2) X(i3) X(s1) X(s2) X(s3) X(fov_h_deg) X(fov_v_deg) \
                      X(cam_h) X(size_k) X(smooth)
static void cfg_load() {
  prefs.begin("duckcam", true);
#define LD(f) vis.cfg.f = prefs.getFloat(#f, vis.cfg.f);
  CFG_FIELDS(LD)
#undef LD
  vis.cfg.flip_x = prefs.getInt("flip_x", vis.cfg.flip_x);
  vis.cfg.flip_y = prefs.getInt("flip_y", vis.cfg.flip_y);
  vis.cfg.lag = prefs.getInt("lag", vis.cfg.lag);
  vis.cfg.rot = prefs.getInt("rot", vis.cfg.rot);
  vis.cfg.res = prefs.getInt("res", vis.cfg.res);
  prefs.end();
}
static void cfg_save() {
  prefs.begin("duckcam", false);
#define SV(f) prefs.putFloat(#f, vis.cfg.f);
  CFG_FIELDS(SV)
#undef SV
  prefs.putInt("flip_x", vis.cfg.flip_x); prefs.putInt("flip_y", vis.cfg.flip_y); prefs.putInt("lag", vis.cfg.lag); prefs.putInt("rot", vis.cfg.rot); prefs.putInt("res", vis.cfg.res);
  prefs.end();
}

// ------------------------------------------------------------------ camera
static bool cam_init() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0; c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM; c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_GRAYSCALE;
  c.frame_size = FRAMESIZE_QVGA;            // buffers voor 320x240 reserveren, zodat "res" live kan wisselen
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t* s = esp_camera_sensor_get();
  s->set_gain_ctrl(s, 1); s->set_exposure_ctrl(s, 1);   // automatisch; bij flikkerende lampen: aec uit
  s->set_framesize(s, vis.cfg.res ? FRAMESIZE_QVGA : FRAMESIZE_QQVGA);   // 320x240 of 160x120 (instelling "res")
  sensor_pid = s->id.PID;
  return true;
}

// ------------------------------------------------------------------ webpagina
static const char PAGE[] PROGMEM = R"HTML(<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Duckling-cam</title><style>body{font-family:sans-serif;margin:12px;background:#111;color:#eee}canvas{width:100%;max-width:480px;image-rendering:pixelated;border:1px solid #444}
.r{display:flex;align-items:center;gap:8px;margin:4px 0}.r label{width:110px}.r input{flex:1}button{margin:4px;padding:8px 12px}pre{background:#222;padding:8px}</style></head><body>
<h3>Duckling-cam</h3><canvas id=c width=80 height=60></canvas><div><button onclick="m=1-m">beeld / verschil</button>
<button onclick="fetch('/save').then(()=>alert('bewaard'))">bewaar</button> <a href="/update" style="color:#8cf">firmware bijwerken</a></div><pre id=s></pre><div id=f></div>
<script>let m=0;const F=[["diff_thr",2,60,1],["min_energy",1,40,0.5],["i1",1,60,0.5],["i2",1,80,0.5],["i3",1,120,0.5],["s1",0,0.2,0.005],["s2",0,0.5,0.005],["s3",0,0.8,0.01],
["smooth",0.05,1,0.05],["lag",1,3,1],["fov_h_deg",30,90,1],["fov_v_deg",20,70,1],["cam_h",0.05,0.3,0.005],["size_k",0.02,0.5,0.01],["rot",0,270,90],["res",0,1,1],["flip_x",0,1,1],["flip_y",0,1,1]];
const c=document.getElementById('c'),x=c.getContext('2d');
fetch('/cfg').then(r=>r.json()).then(g=>{let h='';for(const[k,a,b,st]of F)h+=`<div class=r><label>${k} <b id=v_${k}>${g[k]}</b></label><input type=range min=${a} max=${b} step=${st} value=${g[k]} oninput="document.getElementById('v_${k}').textContent=this.value;fetch('/set?${k}='+this.value)"></div>`;document.getElementById('f').innerHTML=h});
async function tick(){try{const r=await fetch('/frame?d='+m);const b=new Uint8Array(await r.arrayBuffer());const im=x.createImageData(80,60);
for(let i=0;i<4800;i++){const v=m?Math.min(255,b[i]*4):b[i];im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=v;im.data[4*i+3]=255}x.putImageData(im,0,0);
const s=await (await fetch('/stat')).json();if(s.vis){x.strokeStyle='#f40';x.lineWidth=1;x.strokeRect(s.x0+.5,s.y0+.5,s.x1-s.x0-1,s.y1-s.y0-1);x.fillStyle='#0f0';x.fillRect(s.cx-1,s.cy-1,2,2)}
document.getElementById('s').textContent=`ziet iets: ${s.vis?'JA':'nee'}   ${s.fps.toFixed(0)} beelden/s   RP: ${s.rp?'verbonden':'GEEN'}   lamp ${s.lamp} %\nsensor ${({38:'OV2640',13920:'OV3660',22080:'OV5640'})[s.pid]||('0x'+s.pid.toString(16))}, opname ${s.w} px breed\nrichting ${s.bx.toFixed(0)} gr / ${s.by.toFixed(0)} gr (+ = links / omhoog)\ngrootte ${s.msize}  (oppervlak ${(s.area*100).toFixed(1)} %)\nintensiteit ${s.mint}  (gem. verschil ${s.inten.toFixed(1)})\nafstand ${s.dist.toFixed(2)} m\neigen beweging ${s.tflow.toFixed(2)} rad/s   nadering ${s.approach.toFixed(2)} m/s\neigen verschuiving ${s.sx}, ${s.sy} px`}catch(e){}
setTimeout(tick,150)}tick()</script></body></html>)HTML";

static void wifi_set(bool on) {
  if (on == wifi_on) return;
  if (on) {
    WiFi.mode(WIFI_AP); WiFi.softAP(AP_NAAM, AP_WW);
    if (!ota_started) {   // draadloos updaten: Arduino IDE -> Poort -> "duckling-cam at 192.168.4.1"
      ArduinoOTA.setHostname("duckling-cam");
      ArduinoOTA.setPassword(AP_WW);
      ArduinoOTA.onStart([] { ota_busy = true; });   // (de vision-taak pauzeert)
      ArduinoOTA.onEnd([] { ota_busy = false; });
      ArduinoOTA.onError([](ota_error_t) { ota_busy = false; });
      ArduinoOTA.begin(); ota_started = true;
    }
  }
  else { WiFi.softAPdisconnect(true); WiFi.mode(WIFI_OFF); }
  wifi_on = on;
  Serial.printf("wifi %s\n", on ? "AAN" : "UIT");
}
static void web_init() {
  web.on("/", [] { web.send_P(200, "text/html", PAGE); });
  web.on("/frame", [] {
    static uint8_t buf[VW * VH];
    bool d = web.hasArg("d") && web.arg("d") == "1";
    xSemaphoreTake(mtx, portMAX_DELAY); memcpy(buf, d ? snap_diff : snap_gray, sizeof(buf)); xSemaphoreGive(mtx);
    web.send_P(200, "application/octet-stream", (const char*)buf, sizeof(buf));
  });
  web.on("/stat", [] {
    VisionOut o; xSemaphoreTake(mtx, portMAX_DELAY); o = snap_out; xSemaphoreGive(mtx);
    char s[420];
    snprintf(s, sizeof(s), "{\"vis\":%d,\"bx\":%.1f,\"by\":%.1f,\"dist\":%.2f,\"msize\":%d,\"mint\":%d,\"tflow\":%.2f,\"approach\":%.2f,"
             "\"area\":%.4f,\"inten\":%.2f,\"x0\":%d,\"y0\":%d,\"x1\":%d,\"y1\":%d,\"cx\":%.1f,\"cy\":%.1f,\"sx\":%d,\"sy\":%d,\"fps\":%.1f,\"rp\":%d,\"lamp\":%d,\"pid\":%u,\"w\":%d}",
             o.vis, o.bx * 57.3f, o.by * 57.3f, o.dist, o.msize, o.mint, o.tflow, o.approach, o.area, o.inten,
             o.x0, o.y0, o.x1, o.y1, o.cx, o.cy, o.sx, o.sy, fps, (int)(millis() - head_t < 500), (int)(lamp_now == 255 ? 0 : lamp_now), (unsigned)sensor_pid, last_w);
    web.send(200, "application/json", s);
  });
  web.on("/cfg", [] {
    String s = "{";
#define JS(f) s += "\"" #f "\":" + String(vis.cfg.f, 3) + ",";
    CFG_FIELDS(JS)
#undef JS
    s += "\"flip_x\":" + String(vis.cfg.flip_x) + ",\"flip_y\":" + String(vis.cfg.flip_y) + ",\"lag\":" + String(vis.cfg.lag) + ",\"rot\":" + String(vis.cfg.rot) + ",\"res\":" + String(vis.cfg.res) + "}";
    web.send(200, "application/json", s);
  });
  web.on("/set", [] {
    for (int i = 0; i < web.args(); i++) {
      String k = web.argName(i); float v = web.arg(i).toFloat();
#define ST(f) if (k == #f) vis.cfg.f = v;
      CFG_FIELDS(ST)
#undef ST
      if (k == "flip_x") vis.cfg.flip_x = (int)v;
      if (k == "flip_y") vis.cfg.flip_y = (int)v;
      if (k == "lag") vis.cfg.lag = (int)v;
      if (k == "rot") { vis.cfg.rot = (int)v; vis.relight(); }
      if (k == "res" && (int)v != vis.cfg.res) {   // opnameformaat wisselen (live)
        vis.cfg.res = (int)v; sensor_t* sn = esp_camera_sensor_get();
        if (sn) sn->set_framesize(sn, vis.cfg.res ? FRAMESIZE_QVGA : FRAMESIZE_QQVGA);
        vis.relight();
      }
    }
    web.send(200, "text/plain", "ok");
  });
  web.on("/save", [] { cfg_save(); web.send(200, "text/plain", "bewaard"); });
  web_updater.setup(&web, "/update", "duck", AP_WW);   // gebruiker duck, wachtwoord duckling1
  web.begin(); web_started = true;
}

// ------------------------------------------------------------------ UART
static void uart_poll() {
  RpPacket p;
  while (Serial2.available()) {
    if (rp_reader.feed((uint8_t)Serial2.read(), p)) {
      head.yaw = p.head_yaw_mrad * 1e-3f; head.elev = p.head_elev_mrad * 1e-3f;
      head.yaw_rate = p.head_yaw_rate_mrad_s * 1e-3f; head.gyro_z = p.gyro_z_mrad_s * 1e-3f;
      head.fix = p.flags & 1; rp_wifi = (p.flags >> 1) & 1; rp_lamp = p.lamp > 100 ? 100 : p.lamp; head_t = millis();
    }
  }
}
static int16_t clamp16(float v) { return (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v); }
static void send_packet(const VisionOut& o) {
  CamPacket p;
  p.h0 = CAM_H0; p.h1 = CAM_H1; p.seq = seq++;
  p.flags = o.vis ? 1 : 0;
  p.bx_mrad = clamp16(o.bx * 1000); p.by_mrad = clamp16(o.by * 1000);
  p.dist_mm = (uint16_t)(o.dist * 1000);
  p.msize = (uint8_t)o.msize; p.mint = (uint8_t)o.mint;
  p.tflow_mrad_s = clamp16(o.tflow * 1000); p.approach_mm_s = clamp16(o.approach * 1000);
  p.fps = (uint8_t)(fps > 255 ? 255 : fps);
  p.sum = duck_sum(&p, sizeof(p) - 1);
  Serial2.write((const uint8_t*)&p, sizeof(p));
}

// ------------------------------------------------------------------ beeldverwerking (eigen taak, core 1)
static void vision_task(void*) {
  uint32_t last_send = 0, n = 0, t_fps = millis();
  for (;;) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(5); continue; }
    uart_poll();
    if (millis() - head_t > 500) { head = HeadInfo(); rp_lamp = 0; }   // geen RP: kop recht, lamp uit
    if (ota_busy) { esp_camera_fb_return(fb); delay(20); continue; }   // tijdens draadloos updaten niets doen
    // flitsled: helderheid van de RP2350; bij een verandering even niets zien (belichting past zich aan)
    if (rp_lamp != lamp_now) {
      int d = (int)rp_lamp - (int)lamp_now;
      if (lamp_now == 255 || d > 3 || d < -3 || rp_lamp == 0) led_hold = LED_HOLD;
      lamp_now = rp_lamp;
      ledcWrite(FLASH_LED, (uint32_t)lamp_now * 255 / 100);
    }
    if (led_hold > 0) { led_hold--; vis.relight(); }
    last_w = fb->width;
    vis.process(fb->buf, fb->width, fb->height, millis() * 1e-3f, head);
    xSemaphoreTake(mtx, portMAX_DELAY);
    memcpy(snap_gray, vis.cur, sizeof(snap_gray)); memcpy(snap_diff, vis.diff, sizeof(snap_diff)); snap_out = vis.out;
    xSemaphoreGive(mtx);
    esp_camera_fb_return(fb);
    n++;
    if (millis() - t_fps >= 1000) { fps = n * 1000.0f / (millis() - t_fps); n = 0; t_fps = millis(); }
    if (millis() - last_send >= 1000 / SEND_HZ) { last_send = millis(); send_packet(vis.out); }
  }
}

void setup() {
  ledcAttachChannel(FLASH_LED, 5000, 8, LED_CH); ledcWrite(FLASH_LED, 0);   // flitsled uit, dimbaar
  Serial.begin(115200);
  Serial2.begin(UART_BAUD, SERIAL_8N1, UART_RX, UART_TX);
  mtx = xSemaphoreCreateMutex();
  vis.reset(); cfg_load();
  if (!cam_init()) { Serial.println("camera start niet - kabeltje/voeding controleren"); delay(2000); ESP.restart(); }
  if (WIFI_AAN) { wifi_set(true); web_init(); }
  Serial.printf("Duckling-cam klaar. Wifi %s, http://192.168.4.1\n", WIFI_AAN ? AP_NAAM : "uit");
  xTaskCreatePinnedToCore(vision_task, "vision", 16384, nullptr, 2, nullptr, 1);
}

void loop() {
  if (WIFI_AAN) {
    bool rp = millis() - head_t < 1500;        // RP2350 verbonden?
    wifi_set(rp ? rp_wifi : true);             // zonder RP2350: altijd aan (afstellen op tafel)
    if (wifi_on && web_started) web.handleClient();
    if (wifi_on && ota_started) ArduinoOTA.handle();
  }
  delay(2);
}

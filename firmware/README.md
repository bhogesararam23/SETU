# SETU node — hardware

**Status: designed and priced, not yet built.** No physical node has been
assembled. The firmware in `setu_node/` is written against the pin map and the
library APIs below but has not been compiled or flashed, and nothing in this
repository should be read as claiming otherwise.

The reason the repository still carries the firmware is that the size-fraction
channels the attribution layer depends on are a **hardware** decision, not a
software one. Whether the coarse mode can be separated at all is settled by which
sensor is on the mast, so the constraint belongs in the repo next to the code that
depends on it.

## Bill of materials

Prices are Indian vendor quotations seen in October 2026, GST-exclusive unless
noted. They are the actual figures a buyer would start from, not estimates, but
they move and they were not re-quoted for this document.

### Full configuration — ₹28,000–40,000 per node

| Part | Purpose | ₹ |
| --- | --- | --- |
| Sensirion SPS30 | the only common low-cost unit with four mass channels PM1/2.5/4/10 — this is what makes the coarse fraction measured rather than inferred | 3,099 |
| Plantower PMS5003 | three mass bins plus six raw number counts, 0.3–10 µm — the number-size distribution | 1,514–1,930 |
| Bosch BME280 | temperature, humidity, pressure — input to the hygroscopic correction | 284 |
| ESP32-S3-DevKitC-1-N8 | compute, Wi-Fi/BLE, edge inference | 1,101 |
| Heltec WiFi LoRa 32 V3 | 1–3 km link without a SIM per pole | 2,149–2,285 |
| SIMCom SIM7600CE | 4G backhaul where LoRa cannot reach | 3,000 |
| NEO-6M GPS | position and time, so a ticket can be geolocated and defended | 249 |
| RS485 ultrasonic anemometer | wind vector — the difference between a measurement and an attribution | 15,999 |
| SGP40 / BME688 | MOx VOC, corroborating channel | not quoted |
| Alphasense electrochemical CO | combustion tracer | not quoted |
| Li-ion cell + solar charge management, IP65 enclosure, mounts, cable glands | — | not quoted |

Shared: **8-channel LoRaWAN gateway ₹35,000**, one per 1–3 km cell.

### Basic configuration — ₹6,000–8,500

ESP32 + PMS5003 + BME280 + GPS + supply + enclosure. This build gets a two-bin
PM split and humidity, which is enough for the coarse/fine ratio but **not** for
the number-size discrimination the classifier uses to separate burning from
dust. It is a coverage build, not an attribution build.

The two totals are our own arithmetic from the prices above, not a quotation.

## Wiring

```
                ┌──────────────── ESP32-S3-DevKitC-1 ────────────────┐
 SPS30   TX ────┤ GPIO18 (UART2 RX)              UART2 TX GPIO17 ───────► SPS30 RX
                │                                                    │
 PMS5003 TX ──── GPIO 8 (UART1 RX)              UART1 TX GPIO 9 ───────► PMS5003 RX
                │                                                    │
 BME280  SDA ───┤ GPIO21 (I2C SDA)   SCL GPIO20 ├────────────────────┼───► BME280 SCL
                │        @ 0x76                                      │
 Anem.   RX ──── GPIO11 (UART0 alt) TX GPIO10 ├─► MAX485 RO/DI     │
   DE/RE ───────┤ GPIO 4                         │   (half-duplex Modbus RTU)
                │                                                    │
 STATUS  ─────── GPIO 2                         TAMPER ──► GPIO 6  ├── reed switch
                └────────────────────────────────────────────────────┘
```

Power: 5 V from the signal-pole supply where available, otherwise a 18650 with
MPPT charge from a 6 W panel. The SPS30 fan draws ~420 mA at start-up, which is
the sizing constraint on the solar cell, not the ESP32.

## Sampling and reporting

- Sample every 60 s, report every **300 s**. The 5-minute cadence is not a round
  number chosen for tidiness: water suppression on an arterial is effective for
  about 62 minutes (`J. Environ. Sci.`, 2023, doi:10.1016/j.jes.2021.12.005), so
  an hourly network cannot observe the lever the municipality is pulling.
- `coarse = PM10 − PM2.5` is computed **on the device**, from one sample volume.
  Recomputed in the cloud from separately-reported channels it inherits the
  reporting jitter of both.
- A local hygroscopic correction is applied and the raw value is sent alongside
  it. The correction is advisory; the authoritative calibration is fitted on the
  server against a reference.
- 48-batch store-and-forward ring buffer. A node on a pole with a flaky link is
  the normal case, and dropping readings is how you end up with a network that
  reports nothing during the one episode anyone cared about.

## Health flags, and why they matter more than the values

`health` is a bitmap: `SPS_OK`, `PMS_OK`, `BME_OK`, `WIND_OK`, `TAMPER`,
`LOW_POWER`. The server keys quarantine off these rather than off the readings,
because **a dead optical bench reports a plausible zero** and no value-based
check will catch it. This is the single most common field failure and the one a
mean-based QC pipeline misses completely.

Two more failures the flags are there to catch, both of which the digital twin
injects deliberately so the pipeline is tested against them:

- **Inlet obstruction** (tree growth, a parapet, a bird). Reads low with
  elevated variance. Caught by comparing a node against its upwind neighbours.
- **Gain drift** (fan bearing wear, optical contamination). A slow monotone
  ramp. Caught by the CUSUM on the calibration residual, which is why the
  residual is centred on its in-control mean: a node that is stably 1.5 µg/m³
  high is mis-calibrated, not drifting, and quarantining it loses a sensor that
  is still reporting change correctly.

## Known gaps in the firmware

- `readBme()` and `readWind()` are stubs with the register and Modbus sequences
  described in comments. The bench implementation exists but has not been
  folded back into this tree.
- `client.setInsecure()` is a placeholder. Cert pinning against the ingest
  endpoint is required before anything is deployed on public infrastructure.
- Deep sleep is disabled so the serial console works on the bench. It is the
  first thing that has to change for battery operation.
- Nothing here has been compiled. Treat every line as unverified until it
  flashes and reports.

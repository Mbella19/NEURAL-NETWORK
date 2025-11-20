"""FastAPI dashboard that streams candles + predictions over websockets."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Sequence

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from .feature_builder import prepare_feature_frame
from .feature_stats import load_feature_stats
from .model_loader import load_multitask_model
from .predictor import MultiTaskPredictor

DEFAULT_TESTING_CSV = Path(
    "/Users/gervaciusjr/Desktop/AI Trading Bot/Testing Data/EURUSD_M1_202501020001_202510292358.csv"
)
DEFAULT_CHECKPOINT = Path("checkpoints/multitask/multitask_tft.pt")
DEFAULT_TASKS = (
    "Phase1DirectionTask",
    "Phase2IndicatorTask",
    "Phase3StructureTask",
    "Phase4SmartMoneyTask",
    "Phase5CandlestickTask",
    "Phase6SupportResistanceTask",
    "Phase7AdvancedSMTask",
    "Phase8RiskTask",
    "Phase9IntegrationTask",
)


@dataclass
class LiveDashboardConfig:
    testing_csv: Path = DEFAULT_TESTING_CSV
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    sequence_length: Optional[int] = None
    tasks: Sequence[str] = DEFAULT_TASKS
    stream_delay: float = 0.5
    loop_stream: bool = True


@dataclass
class DashboardState:
    records: List[Dict[str, object]]
    predictor: MultiTaskPredictor
    config: LiveDashboardConfig


HTML_TEMPLATE = Template(
    """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>AI Trading Dashboard</title>
    <script src=\"https://cdn.plot.ly/plotly-2.26.0.min.js\"></script>
    <style>
      body { font-family: sans-serif; margin: 0; background-color: #0f1116; color: #f5f5f5; }
      header { padding: 1rem 2rem; border-bottom: 1px solid #1f2230; }
      #chart { width: 100vw; height: 80vh; }
      .status { padding: 0 2rem 1rem; color: #a0aec0; font-size: 0.9rem; }
    </style>
  </head>
  <body>
    <header>
      <h2>Live EURUSD Predictions</h2>
      <p>Primary task: <strong>$primary_task</strong></p>
    </header>
    <div id="chart"></div>
    <div class="status" id="status">Connecting...</div>
    <section id="predictions-panel">
      <h4>Task Predictions</h4>
      <div id="prediction-rows"></div>
    </section>
    <script>
      const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
      const maxPoints = 400;
      const timestamps = [];
      const opens = [], highs = [], lows = [], closes = [];
      const taskNames = $task_names;
      const predictionSeries = taskNames.reduce((acc, name) => {
        acc[name] = [];
        return acc;
      }, {});
      const predictionLabels = {};
      const rowsContainer = document.getElementById('prediction-rows');
      taskNames.forEach((name) => {
        const row = document.createElement('div');
        row.className = 'prediction-row';
        const label = document.createElement('span');
        label.textContent = name;
        const value = document.createElement('span');
        value.textContent = '-';
        row.appendChild(label);
        row.appendChild(value);
        rowsContainer.appendChild(row);
        predictionLabels[name] = value;
      });

      const layout = {
        paper_bgcolor: '#0f1116',
        plot_bgcolor: '#0f1116',
        font: {color: '#e2e8f0'},
        xaxis: {title: 'Timestamp'},
        yaxis: {title: 'Price', rangemode: 'normal'},
        yaxis2: {
          title: 'Prediction',
          overlaying: 'y',
          side: 'right',
          range: [0, 1],
          showgrid: false
        },
        margin: {t: 50, r: 50, b: 40, l: 60}
      };

      const traces = [
        {
          type: 'candlestick',
          x: timestamps,
          open: opens,
          high: highs,
          low: lows,
          close: closes,
          name: 'EURUSD',
          increasing: {line: {color: '#16c784'}},
          decreasing: {line: {color: '#ea3943'}}
        }
      ];
      taskNames.forEach((name, idx) => {
        traces.push({
          yaxis: 'y2',
          type: 'scatter',
          mode: 'lines',
          x: timestamps,
          y: predictionSeries[name],
          line: {width: 2},
          name: name,
          visible: idx === 0 ? true : 'legendonly'
        });
      });

      Plotly.newPlot('chart', traces, layout, {responsive: true});

      ws.onopen = () => {
        document.getElementById('status').innerText = 'Connected';
      };

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        timestamps.push(payload.timestamp);
        opens.push(payload.open);
        highs.push(payload.high);
        lows.push(payload.low);
        closes.push(payload.close);
        taskNames.forEach((name) => {
          const val = payload.predictions && name in payload.predictions ? payload.predictions[name] : null;
          predictionSeries[name].push(val);
          if (predictionLabels[name]) {
            predictionLabels[name].textContent = val === null || Number.isNaN(val) ? '-' : val.toFixed(3);
          }
        });

        if (timestamps.length > maxPoints) {
          timestamps.shift(); opens.shift(); highs.shift(); lows.shift(); closes.shift();
          taskNames.forEach((name) => predictionSeries[name].shift());
        }

        const xUpdate = [timestamps];
        const yUpdate = [null];
        taskNames.forEach(() => {
          xUpdate.push(timestamps);
        });
        taskNames.forEach((name) => {
          yUpdate.push(predictionSeries[name]);
        });

        Plotly.update('chart', {
          x: xUpdate,
          open: [opens],
          high: [highs],
          low: [lows],
          close: [closes],
          y: yUpdate
        });
      };

      ws.onclose = () => {
        document.getElementById('status').innerText = 'Disconnected';
      };
    </script>
  </body>
</html>
"""
)


def _summarize_record(record: Dict[str, object], primary_task: str) -> Dict[str, object]:
    prediction = 0.0
    preds = record.get("predictions", {})
    clean_preds: Dict[str, float] = {}
    if isinstance(preds, dict):
        for key, value in preds.items():
            try:
                clean_preds[key] = float(value)
            except (TypeError, ValueError):
                clean_preds[key] = float("nan")
        fallback = next((v for v in clean_preds.values() if v == v), 0.0)
        prediction = clean_preds.get(primary_task, fallback)
    return {
        "timestamp": record["timestamp"],
        "open": record["open"],
        "high": record["high"],
        "low": record["low"],
        "close": record["close"],
        "prediction": prediction,
        "predictions": clean_preds,
    }


def create_app(config: LiveDashboardConfig | None = None) -> FastAPI:
    config = config or LiveDashboardConfig()
    app = FastAPI(title="AI Trading Dashboard")
    state: DashboardState | None = None

    def _prepare_state_sync() -> DashboardState:
        logger.info("Loading feature frames from %s", config.testing_csv)
        candles_df, feature_frame = prepare_feature_frame(config.testing_csv)

        logger.info("Loading model from %s", config.checkpoint_path)
        bundle = load_multitask_model(config.checkpoint_path)
        stats = load_feature_stats(bundle.feature_columns)
        predictor = MultiTaskPredictor(
            bundle,
            stats,
            sequence_length=config.sequence_length,
            tasks=config.tasks,
        )
        records = predictor.generate_records(candles_df, feature_frame)
        logger.info("Prepared %s records for streaming", len(records))
        return DashboardState(records=records, predictor=predictor, config=config)

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - runtime initialization
        nonlocal state
        if state is None:
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(None, _prepare_state_sync)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        if state is None:
            return HTMLResponse("<h3>Initializing...</h3>")
        primary = state.config.tasks[0] if state.config.tasks else "Phase1DirectionTask"
        if state.config.tasks:
            task_names = list(state.config.tasks)
        else:
            preds = state.records[0].get("predictions", {}) if state.records else {}
            task_names = list(preds.keys()) if isinstance(preds, dict) else [primary]
        if not task_names:
            task_names = [primary]
        return HTMLResponse(
            HTML_TEMPLATE.substitute(primary_task=primary, task_names=json.dumps(task_names))
        )

    @app.get("/api/records")
    async def list_records(limit: int = 500) -> JSONResponse:
        if state is None:
            return JSONResponse({"status": "initializing"}, status_code=202)
        primary = state.config.tasks[0] if state.config.tasks else "Phase1DirectionTask"
        trimmed = [_summarize_record(rec, primary) for rec in state.records[:limit]]
        return JSONResponse({"count": len(trimmed), "records": trimmed})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        if state is None:
            await websocket.send_text(json.dumps({"status": "initializing"}))
            await websocket.close()
            return

        records = state.records
        primary = config.tasks[0] if config.tasks else "Phase1DirectionTask"
        idx = 0
        try:
            while True:
                record = records[idx]
                payload = _summarize_record(record, primary)
                await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(config.stream_delay)
                idx += 1
                if idx >= len(records):
                    if config.loop_stream:
                        idx = 0
                    else:
                        break
        except WebSocketDisconnect:  # pragma: no cover - user action
            logger.info("WebSocket client disconnected")
        finally:
            await websocket.close()

    return app

import express from "express";

const app = express();

const PORT = 5000;

app.get("/", (_req, res) => {
  res.json({
    service: "RealTimeGuard Backend",
    status: "running",
  });
});

app.listen(PORT, () => {
  console.log(`RealTimeGuard backend running on http://localhost:${PORT}`);
});
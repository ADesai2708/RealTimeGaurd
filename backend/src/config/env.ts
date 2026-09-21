import dotenv from "dotenv";

dotenv.config();

export const config = {
  port: Number(process.env.PORT) || 5000,

  redis: {
    url: process.env.REDIS_URL || "redis://localhost:6379",
  },

  mlService: {
    url: process.env.ML_SERVICE_URL || "http://localhost:8000",
  },
};
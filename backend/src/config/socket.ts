import { Server as HttpServer } from "http";
import { Server } from "socket.io";

let io: Server | null = null;

export function initializeSocket(
  httpServer: HttpServer,
): Server {
  io = new Server(httpServer, {
    cors: {
      origin: "*",
    },
  });

  io.on("connection", (socket) => {
    console.log(
      `Socket client connected: ${socket.id}`,
    );

    socket.on("disconnect", (reason) => {
      console.log(
        `Socket client disconnected: ${socket.id} (${reason})`,
      );
    });
  });

  console.log("Socket.IO server initialized");

  return io;
}

export function getSocketServer(): Server {
  if (!io) {
    throw new Error(
      "Socket.IO server has not been initialized.",
    );
  }

  return io;
}
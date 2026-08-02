import Redis from 'ioredis';

import { REDIS_URL } from './config';

// Singleton Redis client (JWT cache). Lazy reconnects handled by ioredis so
// Next dev hot-reload doesn't blow up the connection.
let client: Redis | null = null;

export function getRedis(): Redis {
  if (client) return client;
  client = new Redis(REDIS_URL, {
    lazyConnect: false,
    maxRetriesPerRequest: 2,
    enableOfflineQueue: false,
  });
  client.on('error', (err) => {
    // Surface but never throw — JWT cache miss is recoverable.
    console.warn('[bff redis] error:', err.message);
  });
  return client;
}

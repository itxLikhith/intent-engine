"""
Real-time Analytics Module

This module implements real-time metrics collection and broadcasting
using WebSockets for live dashboard updates.
"""

import asyncio
import json
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session


class ConnectionManager:
    """Manages WebSocket connections for real-time analytics"""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected.add(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager instance
manager = ConnectionManager()


class RealTimeAnalytics:
    """Handles real-time metrics collection and broadcasting"""

    def __init__(self, db_session: Session):
        self.db = db_session

    async def get_live_metrics(self, campaign_id: int | None = None) -> dict:
        """Get current live metrics"""
        # Import database entities inside the method to avoid circular imports
        from database import Ad as DbAd
        from database import AdGroup as DbAdGroup
        from database import AdMetric as DbAdMetric

        # Get latest metrics from database
        query = self.db.query(DbAdMetric)

        if campaign_id:
            # Join with ad and ad_group to get campaign_id
            query = query.join(DbAd).join(DbAdGroup).filter(DbAdGroup.campaign_id == campaign_id)

        # Get metrics from last hour
        from datetime import timedelta

        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        query = query.filter(DbAdMetric.created_at >= one_hour_ago)

        metrics = query.all()

        # Aggregate metrics
        total_impressions = sum(m.impression_count for m in metrics)
        total_clicks = sum(m.click_count for m in metrics)
        total_conversions = sum(m.conversion_count for m in metrics)

        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        cpc = 0  # Would need cost data
        roas = 0  # Would need revenue data

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "impressions": total_impressions,
            "clicks": total_clicks,
            "conversions": total_conversions,
            "ctr": round(ctr, 2),
            "cpc": round(cpc, 2),
            "roas": round(roas, 2),
            "active_campaigns": self._get_active_campaigns_count(),
            "live_users": self._get_live_users_count(),
        }

    def _get_active_campaigns_count(self) -> int:
        """Get count of currently active campaigns"""
        from database import Campaign as DbCampaign

        return self.db.query(DbCampaign).filter(DbCampaign.status == "active").count()

    def _get_live_users_count(self) -> int:
        """Get estimated count of live users (placeholder)"""
        # In a real implementation, this would track active sessions
        return len(manager.active_connections)


async def handle_analytics_websocket(websocket: WebSocket, db: Session):
    """Handle incoming WebSocket connection for real-time analytics"""
    await manager.connect(websocket)
    analytics = RealTimeAnalytics(db)

    try:
        # Send initial data
        initial_data = await analytics.get_live_metrics()
        await manager.broadcast(initial_data)

        # Send updates every 5 seconds
        while True:
            await asyncio.sleep(5)
            metrics = await analytics.get_live_metrics()
            await manager.broadcast(metrics)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

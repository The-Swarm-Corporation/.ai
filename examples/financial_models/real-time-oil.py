from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger
from pydantic import BaseModel
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import queue
import threading
import time


# Configure loguru logger
logger.add(
    "trading_logs/trading_{time}.log",
    rotation="500 MB",
    compression="zip",
    level="INFO",
    format="{time} {level} {message}",
)


class TradingSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class MarketData:
    """
    Real-time market data structure for oil futures
    """

    timestamp: datetime
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    last_price: Decimal
    volume: int


class Order(BaseModel):
    """
    Order representation with full type validation
    """

    order_id: str
    symbol: str
    side: TradingSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING

    class Config:
        arbitrary_types_allowed = True


class MarketState(BaseModel):
    """
    Current market state including relevant indicators
    """

    symbol: str
    current_price: Decimal
    vwap: Decimal
    volatility: float
    momentum: float
    bid_ask_imbalance: float

    class Config:
        arbitrary_types_allowed = True


class MLFeatures(BaseModel):
    """
    Features used for ML model training and prediction
    """

    price_momentum: float
    volume_momentum: float
    bid_ask_spread: float
    volatility: float
    vwap_ratio: float
    order_imbalance: float


class MarketDataStream:
    """
    Real-time market data handler using Yahoo Finance
    """

    def __init__(self, symbols: List[str], data_queue: queue.Queue):
        """
        Initialize market data stream

        Args:
            symbols: List of symbols to stream (e.g., 'CL=F' for crude oil futures)
            data_queue: Queue to put market data updates into
        """
        self.symbols = symbols
        self.data_queue = data_queue
        self.running = False
        self.tickers = {
            symbol: yf.Ticker(symbol) for symbol in symbols
        }
        self._last_prices = {}
        logger.info(
            f"Initialized market data stream for symbols: {symbols}"
        )

    async def start_streaming(self):
        """
        Start streaming market data
        """
        self.running = True
        logger.info("Starting market data stream")

        while self.running:
            try:
                for symbol, ticker in self.tickers.items():
                    # Get latest data
                    data = ticker.history(period="1d", interval="1m")

                    if len(data) > 0:
                        latest = data.iloc[-1]
                        market_data = MarketData(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            bid=Decimal(
                                str(latest["Close"] * 0.9999)
                            ),  # Approximate bid
                            ask=Decimal(
                                str(latest["Close"] * 1.0001)
                            ),  # Approximate ask
                            bid_size=int(latest["Volume"] // 2),
                            ask_size=int(latest["Volume"] // 2),
                            last_price=Decimal(str(latest["Close"])),
                            volume=int(latest["Volume"]),
                        )

                        self.data_queue.put(market_data)
                        logger.debug(
                            f"Received data for {symbol}: {market_data}"
                        )

                await asyncio.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error in market data stream: {str(e)}")
                await asyncio.sleep(5)  # Back off on error

    def stop_streaming(self):
        """
        Stop the market data stream
        """
        self.running = False
        logger.info("Stopped market data stream")


class RealTimeMLModel:
    """
    Real-time machine learning model with continuous training
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.training_data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        if model_path:
            self.load_model(model_path)

    def preprocess_features(
        self, market_state: MarketState
    ) -> np.ndarray:
        """
        Preprocess features for model input
        """
        features = [
            market_state.momentum,
            market_state.volatility,
            market_state.bid_ask_imbalance,
            float(market_state.current_price / market_state.vwap),
        ]
        return np.array(features).reshape(1, -1)

    def predict(self, market_state: MarketState) -> float:
        """
        Make prediction based on current market state
        """
        with self.lock:
            features = self.preprocess_features(market_state)
            scaled_features = self.scaler.transform(features)
            return float(
                self.model.predict_proba(scaled_features)[0][1]
            )

    def add_training_example(
        self, features: Dict[str, Any], label: int
    ):
        """
        Add new training example to the model
        """
        with self.lock:
            self.training_data.append(
                {"features": features, "label": label}
            )

            # Retrain if we have enough new data
            if len(self.training_data) >= 1000:
                self.retrain()

    def retrain(self):
        """
        Retrain the model with accumulated data
        """
        logger.info("Retraining ML model...")

        X = np.array([d["features"] for d in self.training_data])
        y = np.array([d["label"] for d in self.training_data])

        # Update scaler and transform features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Retrain model
        self.model.fit(X_scaled, y)

        # Clear training data
        self.training_data = []

        logger.info("Model retraining completed")


class TradingStrategy:
    """
    High frequency trading strategy implementation
    """

    def __init__(
        self,
        symbols: List[str],
        position_limit: Decimal,
        risk_limit: Decimal,
        model_path: Optional[str] = None,
    ):
        self.symbols = symbols
        self.position_limit = position_limit
        self.risk_limit = risk_limit

        # Initialize components
        self.data_queue: queue.Queue = queue.Queue()
        self.market_data_stream = MarketDataStream(
            symbols, self.data_queue
        )
        self.ml_model = RealTimeMLModel(model_path)

        # Trading state
        self.market_state: Dict[str, MarketState] = {}
        self.positions: Dict[str, Decimal] = {
            sym: Decimal(0) for sym in symbols
        }
        self.pending_orders: List[Order] = []

    async def run(self):
        """
        Main trading loop
        """
        logger.info("Starting trading strategy")

        # Start market data stream
        asyncio.create_task(self.market_data_stream.start_streaming())

        while True:
            try:
                # Process market data
                while not self.data_queue.empty():
                    market_data = self.data_queue.get_nowait()
                    await self.process_market_data(market_data)

                # Trading logic
                for symbol in self.symbols:
                    if symbol in self.market_state:
                        await self.evaluate_trading_signals(symbol)

                await asyncio.sleep(0.1)  # Avoid busy loop

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(1)

    def _calculate_vwap(self, market_data: MarketData) -> Decimal:
        """
        Calculate Volume Weighted Average Price
        """
        symbol = market_data.symbol
        price = float(market_data.last_price)
        volume = float(market_data.volume)

        if volume > 0:
            self.price_history[symbol].append(price)
            self.volume_history[symbol].append(volume)

            prices = np.array(list(self.price_history[symbol]))
            volumes = np.array(list(self.volume_history[symbol]))

            vwap = np.sum(prices * volumes) / np.sum(volumes)
            return Decimal(str(vwap))
        else:
            # If no volume, use last price
            return market_data.last_price

    def _calculate_volatility(self, market_data: MarketData) -> float:
        """
        Calculate rolling volatility
        """
        symbol = market_data.symbol
        price = float(market_data.last_price)
        self.price_history[symbol].append(price)

        if len(self.price_history[symbol]) >= 2:
            returns = (
                np.diff(list(self.price_history[symbol]))
                / np.array(list(self.price_history[symbol]))[:-1]
            )
            return float(
                np.std(returns) * np.sqrt(252)
            )  # Annualized volatility
        return 0.0

    async def process_market_data(self, market_data: MarketData):
        """
        Process incoming market data and update state
        """
        symbol = market_data.symbol

        # Update market state
        self.market_state[symbol] = MarketState(
            symbol=symbol,
            current_price=market_data.last_price,
            vwap=self._calculate_vwap(market_data),
            volatility=self._calculate_volatility(market_data),
            momentum=self._calculate_momentum(market_data),
            bid_ask_imbalance=(
                market_data.bid_size - market_data.ask_size
            )
            / (market_data.bid_size + market_data.ask_size),
        )

    async def evaluate_trading_signals(self, symbol: str):
        """
        Evaluate trading signals and execute trades
        """
        state = self.market_state[symbol]
        prediction = self.ml_model.predict(state)

        # Trading logic based on ML prediction
        if (
            prediction > 0.7
            and self.positions[symbol] < self.position_limit
        ):
            await self.place_order(
                symbol=symbol,
                side=TradingSide.BUY,
                quantity=Decimal("1.0"),
                price=state.current_price,
            )
        elif (
            prediction < 0.3
            and self.positions[symbol] > -self.position_limit
        ):
            await self.place_order(
                symbol=symbol,
                side=TradingSide.SELL,
                quantity=Decimal("1.0"),
                price=state.current_price,
            )

    async def place_order(
        self,
        symbol: str,
        side: TradingSide,
        quantity: Decimal,
        price: Decimal,
    ):
        """
        Place a new order
        """
        order = Order(
            order_id=f"order_{int(time.time()*1000)}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
        )

        logger.info(f"Placing order: {order}")
        self.pending_orders.append(order)

        # In a real system, would interact with broker/exchange here
        # For now, we'll simulate order execution
        await self.simulate_order_execution(order)

    async def simulate_order_execution(self, order: Order):
        """
        Simulate order execution (in real system, would interact with exchange)
        """
        await asyncio.sleep(0.1)  # Simulate network latency

        # Update position
        position_delta = (
            order.quantity
            if order.side == TradingSide.BUY
            else -order.quantity
        )
        self.positions[order.symbol] += position_delta

        # Update order status
        order.status = OrderStatus.FILLED
        logger.info(f"Order filled: {order}")


async def main():
    """
    Main function to run the trading strategy
    """

    # Oil-related symbols
    OIL_SYMBOLS = [
        "BZ=F",  # Brent Crude Oil Futures
        "CL=F",  # WTI Crude Oil Futures
        "USO",  # United States Oil Fund
        "UCO",  # ProShares Ultra Bloomberg Crude Oil
        "XOP",  # SPDR S&P Oil & Gas Exploration & Production ETF
    ]

    # Initialize strategy with oil futures
    strategy = TradingStrategy(
        symbols=OIL_SYMBOLS,
        position_limit=Decimal("10.0"),
        risk_limit=Decimal("100000.0"),
    )

    try:
        await strategy.run()
    except KeyboardInterrupt:
        logger.info("Shutting down trading strategy")
        strategy.market_data_stream.stop_streaming()


if __name__ == "__main__":
    asyncio.run(main())

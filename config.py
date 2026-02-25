"""
Configuration management for the Autonomous Evolutionary Trading System.
Centralizes all configurable parameters with proper type hints and validation.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Configuration for data handling."""
    # Data sources
    EXCHANGE: str = "binance"  # ccxt exchange ID
    SYMBOL: str = "BTC/USDT"
    TIMEFRAME: str = "1h"
    LOOKBACK_PERIOD: int = 1000  # Candles to fetch
    
    # Feature engineering
    INDICATORS: List[str] = None  # Will be initialized in __post_init__
    NORMALIZATION_METHOD: str = "minmax"  # minmax, zscore, robust
    
    def __post_init__(self):
        if self.INDICATORS is None:
            self.INDICATORS = [
                "RSI", "MACD", "BBANDS", "ATR", "OBV",
                "SMA_20", "SMA_50", "EMA_12", "EMA_26"
            ]

@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm evolution."""
    POPULATION_SIZE: int = 100
    GENERATIONS: int = 50
    ELITISM_COUNT: int = 5
    MUTATION_RATE: float = 0.15
    CROSSOVER_RATE: float = 0.7
    
    # Strategy parameters ranges
    STRATEGY_PARAMS_RANGES: Dict = None
    
    def __post_init__(self):
        if self.STRATEGY_PARAMS_RANGES is None:
            self.STRATEGY_PARAMS_RANGES = {
                "rsi_period": (10, 30),
                "rsi_overbought": (60, 80),
                "rsi_oversold": (20, 40),
                "macd_fast": (8, 15),
                "macd_slow": (21, 35),
                "macd_signal": (7, 12),
                "stop_loss_pct": (0.5, 5.0),
                "take_profit_pct": (1.0, 10.0),
                "position_size_pct": (1.0, 20.0)
            }

@dataclass
class RLConfig:
    """Configuration for reinforcement learning component."""
    MODEL_TYPE: str = "DQN"  # DQN, PPO, A2C
    LEARNING_RATE: float = 0.001
    GAMMA: float = 0.99
    MEMORY_SIZE: int = 10000
    BATCH_SIZE: int = 64
    TARGET_UPDATE_FREQ: int = 100
    EXPLORATION_START: float = 1.0
    EXPLORATION_END: float = 0.01
    EXPLORATION_DECAY: float = 0.995

@dataclass
class FirebaseConfig:
    """Configuration for Firebase integration."""
    CREDENTIALS_PATH: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_PATH")
    PROJECT_ID: Optional[str] = os.getenv("FIREBASE_PROJECT_ID")
    COLLECTION_STRATEGIES: str = "trading_strategies"
    COLLECTION_PERFORMANCE: str = "strategy_performance"
    COLLECTION_STATE: str = "system_state"
    
    def validate(self) -> bool:
        """Validate Firebase configuration."""
        if not self.CREDENTIALS_PATH:
            raise ValueError("FIREBASE_CREDENTIALS_PATH environment variable not set")
        if not os.path.exists(self.CREDENTIALS_PATH):
            raise FileNotFoundError(f"Firebase credentials file not found: {self.CREDENTIALS_PATH}")
        return True

@dataclass
class TradingConfig:
    """Configuration for trading execution."""
    PAPER_TRADING: bool = True
    INITIAL_CAPITAL: float = 10000.0
    COMMISSION_PCT: float = 0.1
    SLIPPAGE_PCT: float = 0.05
    MAX_POSITIONS: int = 5
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade

class Config:
    """Main configuration class aggregating all sub-configs."""
    def __init__(self):
        self.data = DataConfig()
        self.ga = GeneticAlgorithmConfig()
        self.rl = RLConfig()
        self.firebase = FirebaseConfig()
        self.trading = TradingConfig()
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

# Global config instance
config = Config()
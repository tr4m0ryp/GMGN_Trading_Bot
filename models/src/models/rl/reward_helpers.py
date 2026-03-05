"""
Reward calculation helpers for the TradingEnvironmentV2 environment.

Extracted from environment.py to keep file sizes under 300 lines.
Contains per-trade reward logic and end-of-episode bonus computation.

Author: Trading Team
Date: 2025-12-24
"""


def calculate_trade_reward(
    trade_return: float,
    hold_duration: int,
    is_win: bool,
    base_trade_reward: float,
    profit_scale: float,
    win_bonus_base: float,
    win_bonus_scale: float,
    win_bonus_cap: float,
    loss_scale: float,
    loss_cap: float,
    quick_trade_threshold: int,
    quick_trade_bonus: float,
) -> float:
    """
    Calculate per-trade reward using the V5.1 reward system.

    For winning trades:
        reward = base + (return * profit_scale) + win_bonus + quick_bonus
        Example: +3% return = 0.3 + 0.9 + 0.8 = +2.0
        Example: +10% return = 0.3 + 3.0 + 1.5 = +4.8

    For losing trades:
        reward = base + max(loss_cap, return * loss_scale)
        Example: -3% loss = 0.3 + max(-0.25, -0.45) = +0.05
        Minimum possible reward: base + loss_cap (always positive with defaults)

    @param trade_return: Return as a fraction (e.g. 0.03 for +3%).
    @param hold_duration: Number of steps the position was held.
    @param is_win: Whether the trade was profitable.
    @param base_trade_reward: Base reward for completing any trade.
    @param profit_scale: Scale factor for profit component.
    @param win_bonus_base: Base bonus for winning trades.
    @param win_bonus_scale: Scale for additional win bonus.
    @param win_bonus_cap: Cap for win bonus.
    @param loss_scale: Scale factor for loss penalty.
    @param loss_cap: Maximum loss penalty (negative value).
    @param quick_trade_threshold: Max steps for quick trade bonus.
    @param quick_trade_bonus: Max bonus for quick profitable trades.

    @return Computed reward for the trade.
    """
    if is_win:
        profit_component = trade_return * profit_scale
        win_bonus = win_bonus_base + min(
            win_bonus_cap,
            trade_return * win_bonus_scale
        )
        reward = base_trade_reward + profit_component + win_bonus

        # Quick trade bonus: faster exits get more bonus
        if hold_duration <= quick_trade_threshold:
            time_factor = 1.0 - (hold_duration / quick_trade_threshold)
            reward += quick_trade_bonus * time_factor
    else:
        loss_penalty = max(loss_cap, trade_return * loss_scale)
        reward = base_trade_reward + loss_penalty

    return reward


def calculate_episode_bonus(
    n_trades: int,
    n_wins: int,
    total_pnl: float,
    min_trades_required: int,
    max_trades_target: int,
    sweet_spot_bonus: float,
    overtrade_penalty_scale: float,
    win_rate_bonuses: dict,
    zero_trade_extra: float,
    missing_trade_penalty: float,
    synergy_scale: float,
) -> float:
    """
    Calculate end-of-episode bonus/penalty based on trading performance.

    Includes sweet spot bonus, win rate bonus, profit magnitude bonus,
    synergy bonus, over-trading penalty, and no-trade catastrophic penalty.

    @param n_trades: Total number of trades completed.
    @param n_wins: Number of winning trades.
    @param total_pnl: Total profit/loss for the episode.
    @param min_trades_required: Minimum trades for sweet spot bonus.
    @param max_trades_target: Maximum trades for sweet spot bonus.
    @param sweet_spot_bonus: Bonus for hitting the trade sweet spot.
    @param overtrade_penalty_scale: Penalty per excess trade beyond threshold.
    @param win_rate_bonuses: Dict mapping win rate thresholds to bonuses.
    @param zero_trade_extra: Extra penalty for zero trades.
    @param missing_trade_penalty: Penalty per missing trade below minimum.
    @param synergy_scale: Scale for win rate + profit synergy bonus.

    @return Computed episode bonus (can be negative for penalties).
    """
    bonus = 0.0

    if n_trades >= min_trades_required:
        # Sweet spot bonus: optimal number of trades
        if min_trades_required <= n_trades <= max_trades_target:
            bonus += sweet_spot_bonus
        elif n_trades <= max_trades_target + 3:
            bonus += sweet_spot_bonus * 0.5
        else:
            excess_trades = n_trades - max_trades_target - 3
            bonus -= 0.2 + (excess_trades * overtrade_penalty_scale)

        # Win rate bonus: tiered system
        current_win_rate = n_wins / n_trades
        for threshold, wr_bonus in sorted(
            win_rate_bonuses.items(), reverse=True
        ):
            if current_win_rate >= threshold:
                bonus += wr_bonus
                break

        # Profit magnitude bonus
        if total_pnl > 0:
            profit_bonus = min(2.0, total_pnl * 20)
            bonus += profit_bonus

            # Synergy bonus: high WR + high profit
            if current_win_rate >= 0.70:
                synergy = current_win_rate * total_pnl * synergy_scale
                bonus += min(2.0, synergy)
    else:
        # Catastrophic penalty for being too passive
        missing_trades = min_trades_required - n_trades
        passive_penalty = missing_trades * missing_trade_penalty

        if n_trades == 0:
            passive_penalty += zero_trade_extra
        elif n_trades == 1:
            passive_penalty += zero_trade_extra * 0.4
        elif n_trades == 2:
            passive_penalty += zero_trade_extra * 0.2

        bonus -= passive_penalty

    return bonus

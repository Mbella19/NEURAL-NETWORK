"""Order management system (Phase 8.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Order:
    id: int
    side: str
    size: float
    price: float
    status: str = "pending"


class OrderManager:
    def __init__(self) -> None:
        self.orders: Dict[int, Order] = {}
        self.next_id = 1

    def place_order(self, side: str, size: float, price: float) -> Order:
        order = Order(id=self.next_id, side=side, size=size, price=price)
        self.orders[self.next_id] = order
        self.next_id += 1
        return order

    def modify_order(self, order_id: int, **kwargs) -> Optional[Order]:
        order = self.orders.get(order_id)
        if not order:
            return None
        for key, value in kwargs.items():
            setattr(order, key, value)
        return order

    def fill_order(self, order_id: int) -> Optional[Order]:
        order = self.orders.get(order_id)
        if order:
            order.status = "filled"
        return order

    def cancel_order(self, order_id: int) -> Optional[Order]:
        order = self.orders.get(order_id)
        if order:
            order.status = "cancelled"
        return order


__all__ = ["Order", "OrderManager"]

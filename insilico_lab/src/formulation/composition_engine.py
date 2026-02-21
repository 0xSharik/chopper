import logging

logger = logging.getLogger(__name__)

class CompositionEngine:
    """Manages molecule quantities, normalization, and safety limits."""
    
    def normalize(self, mixture_list, max_total=50) -> list[dict]:
        """
        Ensure total solute count does not exceed max_total.
        If exceeded, scales counts proportionally.
        """
        if not mixture_list:
            return []
            
        total_count = sum(item['count'] for item in mixture_list)
        
        if total_count <= max_total:
            return mixture_list
            
        logger.warning(f"Total solute count ({total_count}) exceeds limit ({max_total}). Scaling down.")
        
        # Scale proportionally
        scale_factor = max_total / total_count
        scaled_list = []
        for item in mixture_list:
            new_count = max(1, int(item['count'] * scale_factor))
            scaled_list.append({"smiles": item['smiles'], "count": new_count})
            
        # Final adjustment to ensure we are exactly at or under limit due to rounding
        final_total = sum(item['count'] for item in scaled_list)
        if final_total > max_total:
             # Reduce from the largest component
             scaled_list.sort(key=lambda x: x['count'], reverse=True)
             scaled_list[0]['count'] -= (final_total - max_total)
             
        return scaled_list

    def get_summary(self, mixture_list) -> dict:
        """Provide a summary of the composition."""
        total = sum(item['count'] for item in mixture_list)
        if total == 0:
            return {}
            
        return {
            "total_molecules": total,
            "components": len(mixture_list),
            "composition_pct": [
                {"smiles": item['smiles'], "percentage": round((item['count'] / total) * 100, 1)}
                for item in mixture_list
            ]
        }

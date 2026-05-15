import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

interface EditModelNameDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelId: number | null;
  currentName: string | null;
  onSubmit: (id: number, name: string | null) => void;
  isPending: boolean;
}

export function EditModelNameDialog({
  open,
  onOpenChange,
  modelId,
  currentName,
  onSubmit,
  isPending,
}: EditModelNameDialogProps) {
  const [name, setName] = useState('');

  useEffect(() => {
    if (open) setName(currentName ?? '');
  }, [open, currentName]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (modelId == null) return;
    const trimmed = name.trim();
    onSubmit(modelId, trimmed === '' ? null : trimmed);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>モデル名称の編集</DialogTitle>
          <DialogDescription>
            空欄で保存すると名称はクリア（未設定）になります。
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4 py-2">
          <div className="space-y-1">
            <Label htmlFor="model-name">名称</Label>
            <Input
              id="model-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="例: GBDT v3 (2025年データ)"
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isPending}
            >
              キャンセル
            </Button>
            <Button type="submit" disabled={isPending}>
              {isPending ? '保存中…' : '保存'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

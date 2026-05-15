import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface DeleteModelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modelId: number | null;
  modelName: string | null;
  onConfirm: (id: number) => void;
  isPending: boolean;
}

export function DeleteModelDialog({
  open,
  onOpenChange,
  modelId,
  modelName,
  onConfirm,
  isPending,
}: DeleteModelDialogProps) {
  function handleConfirm() {
    if (modelId == null) return;
    onConfirm(modelId);
  }

  const displayLabel = modelName?.trim() ? `「${modelName}」` : `ID ${modelId}`;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>モデルを削除しますか?</DialogTitle>
          <DialogDescription>
            モデル {displayLabel} を削除します。DB 行と
            <code className="mx-1 rounded bg-muted px-1 font-mono text-xs">
              data/models/&lt;ts&gt;/
            </code>
            ディレクトリの両方が削除されます。この操作は取り消せません。
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            キャンセル
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={handleConfirm}
            disabled={isPending}
          >
            {isPending ? '削除中…' : '削除'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

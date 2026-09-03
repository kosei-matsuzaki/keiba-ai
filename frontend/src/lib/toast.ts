/**
 * sonner の toast API をアプリ全体へ再輸出するだけの層。
 *
 *   import { toast } from '@/lib/toast'
 *   toast.success('保存しました')
 *   toast.error('エラーが発生しました')
 *
 * components/ui/ ではなく lib/ に置いてある。中身は関数であってコンポーネントでは
 * なく、hooks から呼ぶため。ui/ に置いていた頃は hooks -> components という
 * 逆向きの依存が 5 本生えていた。画面に出す <Toaster> は components 側にある。
 */
export { toast } from 'sonner';

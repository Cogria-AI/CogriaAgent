'use client';

import { DownloadIcon } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { Button } from '@/components/ui/button';
import type { RendererProps } from './types';

const COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4', '#a855f7'];

type Row = Record<string, string | number>;

export default function ReportRenderer({ args }: RendererProps) {
  const title = String(args.title ?? 'Report');
  const chartType = String(args.chart_type ?? 'bar');
  const data = (Array.isArray(args.data) ? args.data : []) as Row[];
  const xField = typeof args.x_field === 'string' ? args.x_field : Object.keys(data[0] ?? {})[0] ?? 'x';
  const yFields = Array.isArray(args.y_fields)
    ? (args.y_fields as string[])
    : Object.keys(data[0] ?? {}).filter((k) => k !== xField);

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">{title}</h3>
        {chartType === 'table' && (
          <Button variant="outline" size="sm" className="gap-1.5" onClick={() => exportCsv(title, data)}>
            <DownloadIcon className="size-4" /> CSV
          </Button>
        )}
      </div>
      <div className="min-h-0 flex-1">
        {chartType === 'table' ? (
          <Table data={data} />
        ) : (
          <ResponsiveContainer width="100%" height="100%" minHeight={280}>
            {chartType === 'line' ? (
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xField} fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip />
                <Legend />
                {yFields.map((f, i) => (
                  <Line key={f} type="monotone" dataKey={f} stroke={COLORS[i % COLORS.length]} />
                ))}
              </LineChart>
            ) : chartType === 'pie' ? (
              <PieChart>
                <Tooltip />
                <Pie data={data} dataKey={yFields[0]} nameKey={xField} cx="50%" cy="50%" outerRadius={100} label>
                  {data.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            ) : (
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xField} fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip />
                <Legend />
                {yFields.map((f, i) => (
                  <Bar key={f} dataKey={f} fill={COLORS[i % COLORS.length]} />
                ))}
              </BarChart>
            )}
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

function Table({ data }: { data: Row[] }) {
  if (data.length === 0) return <p className="text-muted-foreground text-sm">No data.</p>;
  const cols = Object.keys(data[0]);
  return (
    <div className="overflow-auto">
      <table className="w-full border-separate border-spacing-0 text-sm">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c} className="border-b bg-muted px-2 py-1 text-start font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c} className="border-b px-2 py-1">
                  {String(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function exportCsv(title: string, data: Row[]) {
  if (data.length === 0) return;
  const cols = Object.keys(data[0]);
  const lines = [cols.join(',')];
  for (const row of data) {
    lines.push(cols.map((c) => JSON.stringify(row[c] ?? '')).join(','));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${title.replace(/\s+/g, '_')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
